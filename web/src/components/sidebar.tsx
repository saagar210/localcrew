"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Separator } from "@/components/ui/separator";

const navItems = [
  { href: "/", label: "Home", icon: "H" },
  { href: "/executions", label: "Executions", icon: "E" },
  { href: "/reviews", label: "Reviews", icon: "R" },
  { href: "/workflows", label: "Workflows", icon: "W" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 border-r bg-muted/30 p-4">
      <div className="mb-6">
        <h1 className="text-xl font-bold">LocalCrew</h1>
        <p className="text-sm text-muted-foreground">AI Task Automation</p>
      </div>

      <Separator className="mb-4" />

      <nav className="space-y-1">
        {navItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              "flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
              pathname === item.href
                ? "bg-primary text-primary-foreground"
                : "hover:bg-muted"
            )}
          >
            <span className="flex h-6 w-6 items-center justify-center rounded bg-muted text-xs font-medium">
              {item.icon}
            </span>
            {item.label}
          </Link>
        ))}
      </nav>

      <Separator className="my-4" />

      <div className="text-xs text-muted-foreground">
        <p>API: localhost:8001</p>
        <p>Model: Qwen2.5:14B-Q4</p>
      </div>
    </aside>
  );
}
