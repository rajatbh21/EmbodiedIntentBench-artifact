import json,collections,sys
for d in sys.argv[1:]:
    rows=[json.loads(l) for l in open(d+"/rows.jsonl")]
    for profile, pr in sorted({(r["profile"], r["protocol"]) for r in rows}):
        rs=[r for r in rows if r["profile"]==profile and r["protocol"]==pr]
        if not rs: continue
        c=collections.Counter(); ct=collections.Counter(); proc=collections.defaultdict(list); late=collections.Counter(); en=collections.defaultdict(list)
        cond=collections.defaultdict(collections.Counter)
        for r in rs:
            c[r["policy"]]+=r["success"]
            if r["split"]=="transfer":
                ct[r["policy"]]+=r["success"]; late[r["policy"]]+=r["late"]; en[r["policy"]].append(r["task_energy_j"])
                cond[r["policy"]][(r["link"],r["energy"])]+=r["success"]
            proc[r["policy"]].append(r["processed"])
        print(f"== {profile} {pr}: total {sum(r['success'] for r in rs)}/{len(rs)}")
        for p in sorted(c):
            print(f"  {p:20s} all72={c[p]:2d} tr48={ct[p]:2d} proc={sum(proc[p])/len(proc[p]):5.1f} late48={late[p]:3d} E48={sum(en[p])/len(en[p]):6.1f}J  {dict(cond[p])}")
