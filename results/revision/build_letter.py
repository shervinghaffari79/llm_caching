import re, sys, json
import os
MS=os.environ.get("LETTER_MS","/Users/shervin/Downloads/llm_caching/manuscript_v2/Manuscript-2026-08-09-WithoutAuthorsDetails.tex")
src=open(MS,encoding="utf8").read()
body=src[src.index(r"\begin{document}"):]
nocom="\n".join(re.sub(r"(?<!\\)%.*","",l) for l in body.split("\n"))

# ---- numbering: label -> bare number
sec=[0,0,0]; tab=fig=eq=0; num={}; cur=None; title={}; curt=None
for m in re.finditer(r"\\(section|subsection|subsubsection)\{([^}]*)\}|\\begin\{(table\*?|figure\*?)\}|\\begin\{equation\}|\\label\{([^}]*)\}",nocom):
    if m.group(1):
        lvl={"section":0,"subsection":1,"subsubsection":2}[m.group(1)]
        sec[lvl]+=1
        for i in range(lvl+1,3): sec[i]=0
        cur=".".join(str(x) for x in sec[:lvl+1]); curt=m.group(2)
    elif m.group(3):
        curt=None
        if m.group(3).startswith("table"): tab+=1; cur=str(tab)
        else: fig+=1; cur=str(fig)
    elif m.group(0)=="\\begin{equation}":
        eq+=1; cur=str(eq)
    elif m.group(4):
        num[m.group(4)]=cur
        if curt: title[m.group(4)]=curt

def resolve(t):
    t=re.sub(r"\\ref\{([^}]*)\}",lambda m:num[m.group(1)],t)
    t=re.sub(r"\s*\\cite[pt]?\{[^}]*\}","",t)
    return t

# ---- paragraph extraction from the manuscript
paras=[p.strip() for p in re.split(r"\n\s*\n",nocom)]
def get(par, start=None, end=None):
    c=[p for p in paras if par in p]
    assert len(c)>=1,(par,len(c))
    if len(c)>1: print('NOTE first of',len(c),'matches:',par[:50])
    p=c[0]; a=p.index(start) if start else 0
    b=(p.index(end,a)+len(end)) if end else len(p)
    t=p[a:b]
    t=re.sub(r"\\(?:sub)*section\*?\{[^}]*\}\s*","",t)
    t=re.sub(r"\\label\{[^}]*\}\s*","",t)
    return resolve(re.sub(r"\s*\n\s*"," ",t).strip())
def caption(label):
    i=nocom.index("\\label{%s}"%label)
    st=nocom.rindex("\\caption{",0,i); j=st+len("\\caption{"); d=1
    while d:
        d+= (nocom[j]=="{")-(nocom[j]=="}"); j+=1
    return resolve(nocom[st+len("\\caption{"):j-1])

EXC={
 "r1_1": get("When the cost of a wrong answer is known and labeled traffic",end="(Section~\\ref{subsubsec:latency_breakdown})."),
 "r1_2": get("prices the false hits of each configuration at its default threshold",end="$p\\leq 0.015$)."),
 "r1_3": get("Batching reduces the cost of the cross-encoder considerably"),
 "r1_4a": get("The weights determine which of the three signals",start="The selection places all weight",end="boost-only weights, which are held fixed throughout the evaluation."),
 "r1_4b": get("reports the correct-hit ratio. Semantic Decay is not significantly better",end="Holm $p=0.034$)."),
 "r1_4c": get("Across nine capacities and five streams, the correct-hit ratio of Semantic Decay"),
 "r1_5a": get("varies one generator parameter at a time",start="The ordering of the configurations is the same"),
 "r1_5b": get("Neither workload contains natural multi-turn",start="Neither workload contains natural multi-turn",end="without the dialogue structure it was designed for."),
 "r1_5c": get("Finally, a shared cache can expose information across users"),
 "r2_1a": get("Recent work reduces LLM inference cost through model optimization"),
 "r2_1b": get("These methods raise hit quality by adding signals of increasing cost"),
 "r2_2": caption("fig:system_overview"),
 "r2_3": get("isolates the cost of each component in a test-bench setting"),
 "r2_4": get("reports the five evaluation streams. The four threshold-based rules",start="vCache is the most precise system evaluated"),
 "r3_1a": get("scores the $k$ nearest cached entries of each request",end="mainly adds chances to accept a wrong one."),
 "r3_1b": get("Selective verification targets the requests",start="Verification is inexpensive",end="exactly the decisions of C2."),
 "r3_2a": get("LRU, LeCaR and SphereLFU recover at once",start="Semantic Decay collapses when the cache is small.",end="against 44.9\\%."),
 "r3_2b": get("exceeds W-TinyLFU by 1.87, 1.02 and 0.58 points",start="ReuseGate exceeds W-TinyLFU",end="for all five comparisons)."),
 "r3_2c": get("After an abrupt change of topics, however, it trails LRU",start="After an abrupt change of topics, however,"),
 "r3_3a": get("Each system is reimplemented at the level of its decision rule",start="Each system is reimplemented",end="which entries they keep."),
 "r3_3c": get("Each system is reimplemented at the level of its decision rule",start="ContextCache adds a context-integrated",end="for which it was designed."),
 "r3_3b": get("reports the five evaluation streams. The four threshold-based rules",start="Full threshold sweeps",end="(Section~\\ref{subsubsec:pricing_false_hits})."),
 "r3_4": get("records the cache contents at every eviction at capacity 400",start="Redundancy is low under every policy"),
 "r3_5a": get("serves the seed-42 stream with three bi-encoders"),
 "r3_5b": get("replaces exact search with HNSW",start="Every setting yields the same",end="dominated by the encoder."),
 "r3_6": get("shows that label precision is a close proxy",end="Cohen's $\\kappa=0.40$ over 228 judged pairs."),
}
tpl=open(sys.argv[1] if len(sys.argv)>1 else "letter_template.tex",encoding="utf8").read()
out=re.sub(r"<<EXC:([a-z0-9_]+)>>",lambda m:EXC[m.group(1)],tpl)
out=re.sub(r"<<title:([a-z_:0-9]+)>>",lambda m:title[m.group(1)],out)
out=re.sub(r"<<([a-z_:0-9]+)>>",lambda m:num[m.group(1)],out)
open("Response_to_Reviewers.tex","w",encoding="utf8").write(out)
print("built; tables",tab,"figures",fig,"equations",eq)
for k in ("tab:c1_c2_paired","tab:matched_precision","tab:best_per_lambda","tab:risk_adjusted","tab:latency_breakdown","tab:latency_distribution","tab:cross_batching","tab:topk","tab:gating","tab:workload_sensitivity","tab:cheap_vs_expensive","fig:latency_accuracy","tab:baseline_comparison","tab:answer_audit","tab:encoders","tab:index","tab:eviction_capacity","tab:eviction_density","tab:eviction_shift","tab:learned_admission","tab:related_work","tab:component_activation","fig:system_overview","eq:ras","eq:breakeven","subsec:semantic_decay","subsubsec:eviction_policy_impact","subsubsec:learned_admission","subsubsec:pricing_false_hits","subsubsec:latency_breakdown","subsec:topk_gating","subsubsec:workload_sensitivity","subsec:answer_audit","subsec:encoder_index","subsec:baseline_comparison","subsec:real_query_log","subsubsec:searchqueries_data","sec:limitations","sec:related_work","subsec:ablation","subsec:evaluation_setup"):
    print(k,num.get(k))
