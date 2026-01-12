"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getKASStats, type KASStats } from "@/lib/api";

export function KASStatsWidget() {
  const [stats, setStats] = useState<KASStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    async function fetchKASStats() {
      try {
        const data = await getKASStats();
        if (data) {
          setStats(data);
          setConnected(true);
        } else {
          setConnected(false);
        }
      } catch {
        setConnected(false);
      } finally {
        setLoading(false);
      }
    }

    fetchKASStats();

    // Refresh every 60 seconds
    const interval = setInterval(fetchKASStats, 60000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Card className="border-dashed">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
            <span className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" />
            Knowledge Base
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-6 bg-muted animate-pulse rounded" />
        </CardContent>
      </Card>
    );
  }

  if (!connected || !stats) {
    return (
      <Card className="border-dashed opacity-60">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
            <span className="w-2 h-2 bg-gray-400 rounded-full" />
            Knowledge Base
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">KAS offline</p>
        </CardContent>
      </Card>
    );
  }

  // Calculate type breakdown
  const typeEntries = Object.entries(stats.content_by_type || {});
  const topTypes = typeEntries.slice(0, 3);

  return (
    <Card className="border-primary/20">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
          <span className="w-2 h-2 bg-green-500 rounded-full" />
          Knowledge Base
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-3xl font-bold">{stats.total_content}</div>
        <p className="text-sm text-muted-foreground">
          {stats.total_chunks.toLocaleString()} chunks
        </p>
        {topTypes.length > 0 && (
          <div className="flex gap-1 mt-2 flex-wrap">
            {topTypes.map(([type, count]) => (
              <Badge key={type} variant="secondary" className="text-xs">
                {type}: {count}
              </Badge>
            ))}
          </div>
        )}
        {stats.review_due > 0 && (
          <p className="text-xs text-amber-600 mt-2">
            {stats.review_due} reviews due
          </p>
        )}
      </CardContent>
    </Card>
  );
}
