"use client";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from "recharts";

export function RiskChart({data}:{data:{risk_band:string;count:number}[]}){
  const colors:Record<string,string>={low:"#25c3bd",medium:"#5696ef",high:"#eebc63",critical:"#e97279"};
  return <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><BarChart data={data} margin={{top:18,right:8,left:-22,bottom:4}}><CartesianGrid stroke="#263247" strokeDasharray="3 5" vertical={false}/><XAxis dataKey="risk_band" tick={{fill:"#92a1b7",fontSize:12}} axisLine={false} tickLine={false}/><YAxis tick={{fill:"#718198",fontSize:12}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#202b3d",border:"1px solid #354257",borderRadius:10,color:"#fff"}}/><Bar dataKey="count" radius={[6,6,0,0]} maxBarSize={56}>{data.map((item)=><Cell key={item.risk_band} fill={colors[item.risk_band]}/>)}</Bar></BarChart></ResponsiveContainer></div>;
}
