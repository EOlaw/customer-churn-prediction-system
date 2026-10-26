export function PageHeading({eyebrow,title,description}:{eyebrow:string;title:string;description:string}){
  return <div className="page-heading"><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>;
}
export function Panel({title,subtitle,children,className=""}:{title:string;subtitle?:string;children:React.ReactNode;className?:string}){
  return <section className={`panel ${className}`}><div className="panel-heading"><div><h2>{title}</h2>{subtitle&&<p>{subtitle}</p>}</div></div>{children}</section>;
}
export function Empty({message}:{message:string}){return <div className="empty">{message}</div>}
export function Percent({value}:{value:number|null|undefined}){return <>{value==null?"—":`${(value*100).toFixed(1)}%`}</>}
export function DateLabel({value}:{value:string|null|undefined}){return <>{value?new Date(value).toLocaleString("en-US",{dateStyle:"medium",timeStyle:"short"}):"—"}</>}
export function Interval({value,pct=false}:{value:[number,number]|null|undefined;pct?:boolean}){
  if(!value) return <>—</>;
  const f=(x:number)=>pct?`${(x*100).toFixed(1)}%`:x.toFixed(3);
  return <>[{f(value[0])}, {f(value[1])}]</>;
}
export function Num({value,digits=3}:{value:number|null|undefined;digits?:number}){return <>{value==null?"—":value.toFixed(digits)}</>}
const EVIDENCE:Record<string,{label:string;className:string}>={
  train:{label:"Training · in-sample",className:"ev-insample"},
  calibration:{label:"Calibration · in-sample",className:"ev-insample"},
  test:{label:"Held-out test",className:"ev-heldout"},
  unknown:{label:"No split recorded · unverified",className:"ev-unknown"},
};
export function EvidenceBadge({partition}:{partition:string|null|undefined}){
  const item=EVIDENCE[partition??"unknown"]??EVIDENCE.unknown;
  return <span className={`evidence ${item.className}`}>{item.label}</span>;
}
const KINDS:Record<string,string>={retrospective_batch:"Retrospective batch",retrospective_on_demand:"Retrospective on-demand",legacy_unknown:"Legacy · origin unknown",operational:"Operational"};
export function ScoreKind({kind}:{kind:string|null|undefined}){return <>{kind?(KINDS[kind]??kind):"—"}</>}
