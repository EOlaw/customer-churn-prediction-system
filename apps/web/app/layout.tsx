import "./globals.css";
import { BarChart3, Users, ShieldAlert, Database, Activity, LayoutDashboard, Settings2, ChevronDown } from "lucide-react";
import Link from "next/link";

const links = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/customers", label: "Customer intelligence", icon: Users },
  { href: "/risk", label: "Risk segmentation", icon: ShieldAlert },
  { href: "/models", label: "Model performance", icon: Activity },
  { href: "/data", label: "Data management", icon: Database },
  { href: "/settings", label: "Settings & definitions", icon: Settings2 },
];

export const metadata = { title: "Customer Churn Intelligence", description: "Banking churn intelligence dashboard" };
export const dynamic = "force-dynamic";

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><div className="shell">
    <aside className="sidebar">
      <Link href="/" className="brand"><span className="brand-mark"><BarChart3 size={22}/></span><span>Churn<span className="brand-light">IQ</span></span></Link>
      <div className="workspace"><span className="workspace-icon">CI</span><span><strong>Customer Intelligence</strong><small>Banking workspace</small></span><ChevronDown size={15}/></div>
      <div className="nav-caption">WORKSPACE</div>
      <nav>{links.map(({href,label,icon:Icon}) => <Link href={href} key={href}><Icon size={18}/><span>{label}</span></Link>)}</nav>
      <div className="sidebar-bottom"><div className="status-dot"/> Connected to live API <span className="side-version">v0.1</span></div>
    </aside>
    <div className="main-area"><header className="topbar"><div className="topbar-left"><span className="product-label">CUSTOMER CHURN INTELLIGENCE</span><span className="topbar-divider"/> Banking</div><div className="topbar-right"><span className="live-pill"><span className="status-dot"/> Banking use case</span><div className="avatar">CI</div></div></header><main>{children}</main></div>
  </div></body></html>;
}
