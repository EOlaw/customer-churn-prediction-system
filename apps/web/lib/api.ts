export type Overview = { customers: number; observed_churn_rate: number|null; mean_model_score: number|null; high_risk: number|null; latest_scoring_date: string|null; scored_customers?: number; definition?: string; source_kind?: string|null; model: {id:string; status:string; metrics: Record<string,number>}|null };
export type Risk = {risk_band:string;count:number};
export type Segment = {segment:string;customers:number;observed_rate:number;mean_probability:number|null};
export type Customer = {id:string;external_id:string;attributes:Record<string,string|number>;observed_exit:number|null;prediction:{id:string;probability:number;risk_band:string;scored_at:string;model_id:string}|null};
export type Dataset = {id:string;definition_key:string;sha256:string;rows:number;source_kind:string;contract_version:string;quality_status:string;quality_report_sha256:string;quality_issues:{code:string;severity:string;column:string|null;count:number}[];imported_at:string};
export type Model = {id:string;status:string;metrics:Record<string,number>;dataset_id:string;artifact_sha256:string;lineage:{dataset:{sha256:string;rows:number;source_kind:string;contract_version:string;quality_report_sha256:string}};trained_at:string};

export async function api<T>(path:string):Promise<T>{
  const base=process.env.API_URL ?? "http://localhost:8000";
  const key=process.env.API_KEY;
  if(!key) throw new Error("API_KEY is not configured for the dashboard");
  const res=await fetch(`${base}${path}`,{headers:{"X-API-Key":key},cache:"no-store"});
  if(!res.ok) throw new Error(`API request failed (${res.status})`);
  return res.json() as Promise<T>;
}
