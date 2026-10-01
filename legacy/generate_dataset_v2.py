import json, os
import numpy as np
from tqdm import tqdm
from config import CONFIG

PHENOMENA=["branching","cavitation","porosity","erosion","layering","clustering","fragmentation"]

def grid(n):
    a=np.linspace(-1,1,n,dtype=np.float32)
    return np.meshgrid(a,a,a,indexing="ij")

def norm(x):
    x=x.astype(np.float32); return (x-x.min())/(x.max()-x.min()+1e-8)

def noise(X,Y,Z,rng,lo=1.5,hi=4):
    q=np.zeros_like(X)
    for _ in range(3):
        s=rng.uniform(lo,hi); p=rng.uniform(-np.pi,np.pi,3)
        q+=np.sin(s*X+p[0])*np.cos(.83*s*Y+p[1])*np.sin(.71*s*Z+p[2])
    return norm(q)

def gauss(X,Y,Z,c,s):
    c=np.asarray(c); s=np.maximum(np.asarray(s),1e-4)
    return np.exp(-(((X-c[0])/s[0])**2+((Y-c[1])/s[1])**2+((Z-c[2])/s[2])**2)).astype(np.float32)

def vec(rng):
    v=rng.normal(size=3); return (v/max(np.linalg.norm(v),1e-8)).astype(np.float32)

def segment(X,Y,Z,a,b,r):
    a=np.asarray(a); d=np.asarray(b)-a; q=np.dot(d,d)
    if q<1e-8: dist=np.sqrt((X-a[0])**2+(Y-a[1])**2+(Z-a[2])**2)
    else:
        t=np.clip(((X-a[0])*d[0]+(Y-a[1])*d[1]+(Z-a[2])*d[2])/q,0,1)
        dist=np.sqrt((X-(a[0]+t*d[0]))**2+(Y-(a[1]+t*d[1]))**2+(Z-(a[2]+t*d[2]))**2)
    return np.exp(-(dist/max(r,1e-4))**2)

def base(X,Y,Z,rng):
    c=rng.uniform(-.1,.1,3); s=rng.uniform(.42,.70,3)
    f=gauss(X,Y,Z,c,s)*(.86+.28*noise(X,Y,Z,rng))
    return f>rng.uniform(.30,.43),{"center":c.tolist(),"scale":s.tolist()}

def branching(X,Y,Z,rng):
    fields=[]; meta=[]; depth=int(rng.integers(2,4))
    def grow(a,d,L,r,k):
        d=d+rng.normal(scale=.22,size=3); d/=max(np.linalg.norm(d),1e-8)
        b=np.clip(a+d*L,-.92,.92); fields.append(segment(X,Y,Z,a,b,r))
        meta.append({"depth":k,"start":a.tolist(),"end":b.tolist(),"radius":float(r)})
        if k<depth:
            for _ in range(int(rng.integers(2,4))):
                nd=d+rng.normal(scale=.55,size=3); nd/=max(np.linalg.norm(nd),1e-8)
                grow(b,nd,L*rng.uniform(.62,.88),max(.055,r*rng.uniform(.58,.76)),k+1)
    grow(rng.uniform(-.1,.1,3),vec(rng),rng.uniform(.38,.65),rng.uniform(.10,.17),0)
    return np.max(fields,0)>.42,{"depth":depth,"branches":meta}

def shell(m):
    q=m.copy()
    for ax in range(3): q&=np.roll(m,1,ax)&np.roll(m,-1,ax)
    s=m&~q; s[[0,-1],:,:]=False; s[:,[0,-1],:]=False; s[:,:,[0,-1]]=False
    return s

def cavitation(X,Y,Z,rng):
    m,b=base(X,Y,Z,rng); fs=[]; meta=[]
    for _ in range(int(rng.integers(10,22))):
        c=rng.uniform(-.72,.72,3); surf=rng.random()<.48
        if surf:c=vec(rng)*rng.uniform(.48,.78)
        s=rng.uniform(.07,.22,3); fs.append(gauss(X,Y,Z,c,s)**rng.uniform(.72,1.35))
        meta.append({"center":c.tolist(),"scale":s.tolist(),"surface_bias":bool(surf)})
    m[np.max(fs,0)>rng.uniform(.45,.62)]=False
    return m,{"base":b,"cavity_count":len(fs),"connectivity_bias":"high","cavities":meta}

def porosity(X,Y,Z,rng):
    m,b=base(X,Y,Z,rng); pts=np.argwhere(shell(m)); holes=np.zeros_like(m); meta=[]
    if len(pts):
        for _ in range(int(rng.integers(18,38))):
            q=pts[int(rng.integers(len(pts)))]; c=np.array([X[tuple(q)],Y[tuple(q)],Z[tuple(q)]])
            s=rng.uniform(.07,.15,3); holes|=gauss(X,Y,Z,c,s)>rng.uniform(.50,.67)
            meta.append({"center":c.tolist(),"scale":s.tolist()})
        m[holes]=False
    return m,{"base":b,"pore_count":len(meta),"surface_condition":True,"pores":meta}

def erosion(X,Y,Z,rng):
    m,b=base(X,Y,Z,rng); agent=str(rng.choice(["water","wind"])); d=vec(rng)
    duration=rng.uniform(.08,.92); intensity=np.clip(duration*rng.uniform(.75,1.25),.05,1)
    p=norm(X*d[0]+Y*d[1]+Z*d[2]); n=noise(X,Y,Z,rng,2,5)
    if agent=="water": e=.6*(1-p)+.4*(np.sin(p*np.pi*rng.uniform(3,6)+n*np.pi)+1)/2
    else: e=.65*p+.35*np.abs(np.sin(p*np.pi*rng.uniform(4,8)+n*2))
    e=norm(e)
    for i in range(1+int(duration*3)):
        s=shell(m); m[s&(rng.random(m.shape)<e*intensity*(.72 if i==0 else .20))]=False
    return m,{"base":b,"agent":agent,"duration":float(duration),"intensity":float(intensity),"direction":d.tolist(),"surface_only_primary":True}

def layering(X,Y,Z,rng):
    m,b=base(X,Y,Z,rng); d=vec(rng); c=X*d[0]+Y*d[1]+Z*d[2]; c+= (noise(X,Y,Z,rng,1.2,3)-.5)*rng.uniform(.15,.38)
    f=rng.uniform(3,7); bands=np.sin(c*np.pi*f); l=noise(X,Y,Z,rng,2,5)
    v=m&((bands>rng.uniform(-.35,.1))|(l>rng.uniform(.6,.78))); v|=m&(np.abs(c)<rng.uniform(.35,.6))
    return v,{"base":b,"layer_frequency":float(f),"orientation":d.tolist(),"continuity":"variable","overlap":float(rng.uniform(.25,.85))}

def clustering(X,Y,Z,rng):
    n=int(rng.integers(5,13)); pos=[rng.uniform(-.72,.72,3) for _ in range(n)]; r=[rng.uniform(.08,.16) for _ in pos]
    d=vec(rng); f=norm(X*d[0]+Y*d[1]+Z*d[2]); f=np.clip(f+.4*(noise(X,Y,Z,rng,1.2,2.8)-.5),0,1); attraction=rng.uniform(.1,.55)
    it=int(rng.integers(5,13))
    for _ in range(it):
        new=[]
        for i,p in enumerate(pos):
            ix=np.clip(((p+1)*.5*(len(X)-1)).astype(int),0,len(X)-1)
            r[i]=min(.34,r[i]+.015+.045*float(f[tuple(ix)])+rng.uniform(0,.012))
            if len(pos)>1:
                j=min((j for j in range(len(pos)) if j!=i),key=lambda j:np.linalg.norm(pos[j]-p))
                dv=pos[j]-p; q=np.linalg.norm(dv)
                if q>1e-6:p=p+dv/q*attraction*.018
            new.append(p)
        pos=new
    return np.max([gauss(X,Y,Z,p,(rr,rr,rr)) for p,rr in zip(pos,r)],0)>.42,{"initial_forms":n,"iterations":it,"field_direction":d.tolist(),"attraction":float(attraction),"coalescence_allowed":True}

def fragmentation(X,Y,Z,rng):
    m,b=base(X,Y,Z,rng); ff=np.zeros_like(m,dtype=np.float32); meta=[]
    for _ in range(int(rng.integers(3,8))):
        d=vec(rng); off=rng.uniform(-.45,.45); plane=X*d[0]+Y*d[1]+Z*d[2]-off
        warp=(noise(X,Y,Z,rng,1.5,4)-.5)*rng.uniform(.04,.14); w=rng.uniform(.018,.055)
        ff=np.maximum(ff,np.exp(-((plane+warp)/w)**2)); meta.append({"direction":d.tolist(),"offset":float(off),"width":float(w)})
    strength=rng.uniform(.15,.72); m[m&(ff>rng.uniform(.45,.68))&(rng.random(m.shape)<strength*.58)]=False
    return m,{"base":b,"fracture_count":len(meta),"fracture_intensity":float(strength),"displacement":0.0,"complete_fracture":False,"fractures":meta}

G={"branching":branching,"cavitation":cavitation,"porosity":porosity,"erosion":erosion,"layering":layering,"clustering":clustering,"fragmentation":fragmentation}

def hybrid(X,Y,Z,rng):
    names=list(rng.choice(PHENOMENA,size=int(rng.integers(2,4)),replace=False)); mode=str(rng.choice(["sequential","simultaneous"])); parts=[]
    if mode=="sequential":
        v=None
        for name in names:
            x,p=G[name](X,Y,Z,rng); v=x.copy() if v is None else (v&x if name in {"cavitation","porosity","erosion","fragmentation"} else v|x); parts.append({"phenomenon":name,"parameters":p})
    else:
        xs=[]
        for name in names:
            x,p=G[name](X,Y,Z,rng); xs.append(x.astype(np.float32)); parts.append({"phenomenon":name,"parameters":p})
        v=np.mean(xs,0)>rng.uniform(.30,.48)
    return v,{"mode":mode,"phenomena":names,"processes":parts}

def components(v):
    if not v.any():return 0
    seen=np.zeros_like(v,bool); c=0
    for s in np.argwhere(v):
        s=tuple(map(int,s))
        if seen[s]:continue
        c+=1; stack=[s]; seen[s]=1
        while stack:
            a,b,z=stack.pop()
            for da,db,dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                q=(a+da,b+db,z+dz)
                if all(0<=q[i]<v.shape[i] for i in range(3)) and v[q] and not seen[q]:
                    seen[q]=1;stack.append(q)
    return c

def valid(v,name):
    occ=float(v.mean()); cc=components(v); ok=.015<=occ<=.55
    if name in {"cavitation","porosity","erosion","layering","fragmentation"}:ok &= cc==1
    return ok,{"occupancy":occ,"connected_components":cc}

def main():
    os.makedirs("data",exist_ok=True); n=int(CONFIG.resolution); total=int(CONFIG.dataset_size); X,Y,Z=grid(n); master=np.random.default_rng(int(CONFIG.seed))
    single=int(round(total*.70)); a=PHENOMENA*(single//7)+PHENOMENA[:single%7]+["hybridization"]*(total-single); master.shuffle(a)
    data=np.zeros((total,n,n,n,1),np.float32); meta=[]
    for i,name in enumerate(tqdm(a,desc="Generating")):
        seed=int(master.integers(0,2**32-1)); accepted=False
        for attempt in range(40):
            r=np.random.default_rng(seed+attempt); v,p=hybrid(X,Y,Z,r) if name=="hybridization" else G[name](X,Y,Z,r); ok,st=valid(v,name)
            if ok:accepted=True;break
        data[i,...,0]=v.astype(np.float32); st["validation_exhausted"]=not accepted
        meta.append({"id":i,"seed":seed,"phenomenon":name,"phenomena":p.get("phenomena",[name]),"parameters":p,"validation":st})
    np.savez_compressed("data/procedural.npz",structures=data)
    with open("data/procedural_metadata.json","w",encoding="utf-8") as f:json.dump(meta,f,indent=2)
    np.savez_compressed("data/procedural_metadata.npz",ids=np.arange(total,dtype=np.int32),seeds=np.array([m["seed"] for m in meta],np.uint64),phenomena=np.array([m["phenomenon"] for m in meta]),occupancy=np.array([m["validation"]["occupancy"] for m in meta],np.float32),connected_components=np.array([m["validation"]["connected_components"] for m in meta],np.int32))
    print("Wrote data/procedural.npz",data.shape,"mean occupancy",float(data.mean()))

if __name__=="__main__":main()
