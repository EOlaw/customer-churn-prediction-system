import Link from "next/link";
import { api, Customer } from "@/lib/api";
import { DateLabel, Empty, PageHeading, Percent } from "@/components/ui";

type CustomerList={items:Customer[];page:number;size:number;total:number};
export default async function CustomersPage({searchParams}:{searchParams:Promise<{q?:string;page?:string}>}){
  const params=await searchParams; const q=params.q??""; const page=Math.max(1,Number(params.page)||1);
  const data=await api<CustomerList>(`/api/v1/customers?search=${encodeURIComponent(q)}&page=${page}&size=20`);
  return <><PageHeading eyebrow="INTELLIGENCE / CUSTOMERS" title="Customer intelligence" description="Search banking customers and inspect their latest scored risk."/>
    <div className="table-panel"><div className="table-top"><div><h2>Customer directory</h2><p>{data.total.toLocaleString()} matching records</p></div><form><input name="q" defaultValue={q} placeholder="Search customer ID" aria-label="Search customer ID"/><button>Search</button></form></div>
    {data.items.length?<div className="table-scroll"><table><thead><tr><th>Customer ID</th><th>Geography</th><th>Tenure</th><th>Products</th><th>Observed exit</th><th>Churn probability</th><th>Risk</th><th>Scored</th></tr></thead><tbody>{data.items.map(c=><tr key={c.id}><td><Link href={`/customers/${c.id}`}>{c.external_id}</Link></td><td>{c.attributes.Geography}</td><td>{c.attributes.Tenure} years</td><td>{c.attributes.NumOfProducts}</td><td>{c.observed_exit===1?"Exited":"Retained"}</td><td><Percent value={c.prediction?.probability}/></td><td><span className={`risk-pill ${c.prediction?.risk_band??"unscored"}`}>{c.prediction?.risk_band??"Unscored"}</span></td><td><DateLabel value={c.prediction?.scored_at}/></td></tr>)}</tbody></table></div>:<Empty message="No customers match this search."/>}
    <div className="pagination"><span>Page {page}</span><div>{page>1&&<Link href={`/customers?q=${encodeURIComponent(q)}&page=${page-1}`}>Previous</Link>}{page*data.size<data.total&&<Link href={`/customers?q=${encodeURIComponent(q)}&page=${page+1}`}>Next</Link>}</div></div></div></>;
}
