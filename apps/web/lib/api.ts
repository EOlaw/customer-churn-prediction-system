export type Interval = [number, number] | null;
export type Evidence = {partition:"train"|"calibration"|"test"|"unknown";in_sample:boolean|null};
export type PartitionSummary = Evidence & {customers:number;scored:number;mean_score:number|null;high_risk:number};
export type HeldOut = {rows:number;positives:number;prevalence:number;roc_auc:number;pr_auc:number;brier:number;recall_at_review_capacity:number|null;review_capacity_share:number;precision:number|null;recall:number|null;f1:number|null;confidence_intervals_95:Record<string,Interval>};
export type Overview = {
  customers:number; observed_churn_rate:number|null; definition:string; source_kind:string|null; latest_scoring_date:string|null;
  model:{id:string;status:string;lineage_status:string;evaluation_recorded:boolean;evidence_status:string|null;metrics:Record<string,number>;held_out:HeldOut|null}|null;
  partitions:Record<string,PartitionSummary>|null; score_kinds:Record<string,number>;
  operational_scoring:{available:boolean;reason?:string}; limitations:string[];
};
export type Risk = {risk_band:string;count:number};
export type RiskResponse = {partition:string;model_id:string|null;in_sample:boolean|null;note:string;items:Risk[]};
export type Segment = {segment:string;customers:number;observed_rate:number;heldout_customers:number;heldout_mean_probability:number|null};
export type Prediction = Evidence & {id:string;probability:number;risk_band:string;scored_at:string;model_id:string;score_kind:string};
export type Customer = {id:string;external_id:string;attributes:Record<string,string|number>;observed_exit:number|null;prediction:Prediction|null};
export type Dataset = {id:string;definition_key:string;sha256:string;rows:number;source_kind:string;lineage_status:string;contract_version:string|null;quality_status:string|null;quality_report_sha256:string|null;quality_issues:{code:string;severity:string;column:string|null;count:number}[]|null;imported_at:string};
export type Model = {id:string;status:string;metrics:Record<string,number>;dataset_id:string;artifact_sha256:string|null;lineage_status:string;evaluation_recorded:boolean;lineage:{dataset:{sha256:string;rows:number;source_kind:string;contract_version:string;quality_report_sha256:string}}|null;trained_at:string};

export type Reliability = {bin_low:number;bin_high:number;rows:number;mean_predicted:number|null;observed_rate:number|null;observed_ci95_low:number|null;observed_ci95_high:number|null;sparse:boolean};
export type SubgroupEntry = {group:string;rows:number;positives:number;prevalence:number;selection_rate:number;status:string;reason:string|null;metrics:{roc_auc:number|null;recall:number|null;precision:number|null;confidence_intervals_95:Record<string,Interval>}|null};
export type Variant = {features:string[];excluded:string[];roc_auc:number;roc_auc_ci95:Interval;pr_auc:number;brier:number;recall_at_review_capacity:number|null;f1_at_threshold:number|null;roc_auc_difference_vs_full?:{difference:number;ci95:[number,number]}|null};
export type Evaluation = {
  framing:string; evidence_status:string; limitations:string[];
  split:{rows:Record<string,number>;positives:Record<string,number>;roles:Record<string,string>;random_state:number};
  selection:{data:string;metric:string;rule:string;selected:string;selected_params:Record<string,unknown>;note:string;candidates:Record<string,{cv_pr_auc_mean:number;cv_pr_auc_se:number;configurations_searched:number}>};
  calibration:{data:string;rows:number;positives:number;selected_method:string;cross_fitted_brier:Record<string,number>};
  threshold:{rule:string;value:number;selected_on:string};
  review_capacity:{share:number;note:string};
  held_out_test:HeldOut & {roc_auc:number;at_threshold:{threshold:number;precision:number|null;recall:number|null;f1:number|null;confusion_matrix:{true_negative:number;false_positive:number;false_negative:number;true_positive:number}};at_review_capacity:{reviewed:number;true_positives_found:number;recall:number|null;precision:number;lift:number|null;capacity_share:number};calibration:{brier:number;brier_prevalence_baseline:number;brier_skill_score:number|null;expected_calibration_error:number;calibration_in_the_large:number;calibration_slope:number|null;reliability:Reliability[]};reference:{roc_auc:number;pr_auc:number}};
  sensitivity_analyses:{note:string;variants:Record<string,Variant>};
  subgroups:{note:string;groups:Record<string,SubgroupEntry[]>};
};
export type EvaluationResponse = {id:string;lineage_status:string;evaluation:Evaluation|null;reason:string|null};

export async function api<T>(path:string):Promise<T>{
  const base=process.env.API_URL ?? "http://localhost:8000";
  const key=process.env.API_KEY;
  if(!key) throw new Error("API_KEY is not configured for the dashboard");
  const res=await fetch(`${base}${path}`,{headers:{"X-API-Key":key},cache:"no-store"});
  if(!res.ok) throw new Error(`API request failed (${res.status})`);
  return res.json() as Promise<T>;
}
