import Link from "next/link";
import { ArrowUpRight, CalendarDays, Users, ShieldAlert, Activity, Gauge, Database } from "lucide-react";
import { api, Overview, Risk, Segment } from "@/lib/api";
import { RiskChart } from "@/components/chart";
import { PageHeading, Panel, Empty, Percent, DateLabel } from "@/components/ui";

export default async function OverviewPage(){
  const [overview,risk,segments]=await Promise.all([
    api<Overview>("/api/v1/analytics/overview"),api<Risk[]>("/api/v1/analytics/risk-distribution"),api<Segment[]>("/api/v1/analytics/segments")]);
  const cards=[
    {label:"Total customers",value:overview.customers.toLocaleString(),note:"Imported banking customers",icon:Users},
    {label:"Observed exit rate",value:<Percent value={overview.observed_churn_rate}/>,note:"Source snapshot label",icon:Activity},
    {label:"Mean model score",value:<Percent value={overview.mean_model_score}/>,note:`Retrospective · ${overview.scored_customers??0} scored`,icon:Gauge},
    {label:"High & critical risk",value:overview.high_risk?.toLocaleString()??"—",note:"Configurable probability bands",icon:ShieldAlert},
  ];
  return <><PageHeading eyebrow="ANALYTICS / OVERVIEW" title="Churn analysis" description="A clear view of banking customer exits, model scores, and risk concentration."/>
    <div className="notice"><span className="notice-icon">i</span><span>{overview.source_kind==="synthetic"?"Synthetic demonstration data. ":""}This dataset is cross-sectional. Exit timing and prediction horizon are unknown; scores below are retrospective and uncalibrated.</span></div>
    <div className="kpi-grid">{cards.map(({label,value,note,icon:Icon})=><div className="kpi" key={label}><div className="kpi-top"><span>{label}</span><Icon size={19}/></div><div className="kpi-value">{value}</div><div className="kpi-note">{note}</div></div>)}</div>
    <div className="content-grid"><Panel title="Customer risk distribution" subtitle="Latest score per customer, grouped by risk band" className="wide">{risk.some(r=>r.count>0)?<RiskChart data={risk}/>:<Empty message="No predictions yet. Train, register, and score a model to populate this chart."/>}</Panel>
      <Panel title="Model snapshot" subtitle="Held-out test evaluation"><div className="model-snapshot">{overview.model?<><div className="model-id">{overview.model.id}</div><div className="metric-line"><span>ROC-AUC</span><strong>{overview.model.metrics.roc_auc?.toFixed(3)??"—"}</strong></div><div className="metric-line"><span>PR-AUC</span><strong>{overview.model.metrics.pr_auc?.toFixed(3)??"—"}</strong></div><div className="metric-line"><span>Brier score</span><strong>{overview.model.metrics.brier?.toFixed(3)??"—"}</strong></div><div className="metric-line"><span>Status</span><strong className="status-text">{overview.model.status}</strong></div></>:<Empty message="No registered model yet."/>}</div><Link href="/models" className="panel-link">View model performance <ArrowUpRight size={15}/></Link></Panel></div>
    <div className="content-grid lower"><Panel title="Exit rate by geography" subtitle="Observed labels in the imported banking snapshot" className="wide">{segments.length?<div className="segments">{segments.map(s=><div className="segment-row" key={s.segment}><div><strong>{s.segment}</strong><span>{s.customers.toLocaleString()} customers</span></div><div className="segment-track"><div style={{width:`${s.observed_rate*100}%`}}/></div><strong><Percent value={s.observed_rate}/></strong></div>)}</div>:<Empty message="Import a banking dataset to see geographic segments."/>}</Panel><Panel title="Data freshness" subtitle="Latest operational events"><div className="freshness"><div><Database size={18}/><span>Definition</span><strong>{overview.definition??"—"}</strong></div><div><CalendarDays size={18}/><span>Latest score</span><strong><DateLabel value={overview.latest_scoring_date}/></strong></div></div><Link href="/data" className="panel-link">View dataset lineage <ArrowUpRight size={15}/></Link></Panel></div>
  </>;
}
