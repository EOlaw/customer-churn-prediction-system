export function PageHeading({eyebrow,title,description}:{eyebrow:string;title:string;description:string}){
  return <div className="page-heading"><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>;
}
export function Panel({title,subtitle,children,className=""}:{title:string;subtitle?:string;children:React.ReactNode;className?:string}){
  return <section className={`panel ${className}`}><div className="panel-heading"><div><h2>{title}</h2>{subtitle&&<p>{subtitle}</p>}</div></div>{children}</section>;
}
export function Empty({message}:{message:string}){return <div className="empty">{message}</div>}
export function Percent({value}:{value:number|null|undefined}){return <>{value==null?"—":`${(value*100).toFixed(1)}%`}</>}
export function DateLabel({value}:{value:string|null|undefined}){return <>{value?new Date(value).toLocaleString("en-US",{dateStyle:"medium",timeStyle:"short"}):"—"}</>}
