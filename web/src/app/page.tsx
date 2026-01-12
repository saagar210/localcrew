"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  getHealth,
  getExecutions,
  getReviewStats,
  type Execution,
  type ReviewStats,
  type HealthStatus,
} from "@/lib/api";
import { StatusBadge, formatDuration, formatDate } from "@/lib/formatters";
import { KASStatsWidget } from "@/components/kas-stats-widget";

export default function Home() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [executions, setExecutions] = useState<Execution[]>([]);
  const [reviewStats, setReviewStats] = useState<ReviewStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [healthData, execData, statsData] = await Promise.all([
          getHealth(),
          getExecutions(5),
          getReviewStats(),
        ]);
        setHealth(healthData);
        setExecutions(execData);
        setReviewStats(statsData);
      } catch (err) {
        console.error("Failed to connect to API:", err);
        setError("Failed to connect to API. Is the server running?");
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-48" />
        <div className="grid gap-4 md:grid-cols-3">
          <Skeleton className="h-32" />
          <Skeleton className="h-32" />
          <Skeleton className="h-32" />
        </div>
        <Skeleton className="h-64" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] text-center">
        <h2 className="text-xl font-semibold mb-2">Connection Error</h2>
        <p className="text-muted-foreground mb-4">{error}</p>
        <Button onClick={() => window.location.reload()}>Retry</Button>
      </div>
    );
  }

  const completedExecs = executions.filter((e) => e.status === "completed");
  const avgConfidence =
    completedExecs.length > 0
      ? completedExecs.reduce((acc, e) => acc + (e.confidence_score || 0), 0) /
        completedExecs.length
      : 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-sm text-muted-foreground">
            {health?.app} v{health?.version} • API {health?.status}
          </p>
        </div>
        <div className="flex gap-2">
          <Link href="/workflows">
            <Button>New Task</Button>
          </Link>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Executions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{reviewStats?.total || 0}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Pending Reviews
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {reviewStats?.pending || 0}
            </div>
            {(reviewStats?.pending || 0) > 0 && (
              <Link
                href="/reviews"
                className="text-sm text-primary hover:underline"
              >
                View pending →
              </Link>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Avg Confidence
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {avgConfidence > 0 ? `${(avgConfidence * 100).toFixed(0)}%` : "-"}
            </div>
          </CardContent>
        </Card>

        <KASStatsWidget />
      </div>

      {/* Recent Executions */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Recent Executions</CardTitle>
            <Link href="/executions">
              <Button variant="ghost" size="sm">
                View All
              </Button>
            </Link>
          </div>
        </CardHeader>
        <CardContent>
          {executions.length === 0 ? (
            <p className="text-center text-muted-foreground py-8">
              No executions yet. Start by creating a new task.
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Type</TableHead>
                  <TableHead>Input</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Duration</TableHead>
                  <TableHead>Created</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {executions.map((exec) => (
                  <TableRow key={exec.id}>
                    <TableCell className="font-medium">
                      {exec.crew_type}
                    </TableCell>
                    <TableCell className="max-w-[200px] truncate">
                      {exec.input_text}
                    </TableCell>
                    <TableCell>
                      <StatusBadge status={exec.status} />
                    </TableCell>
                    <TableCell>
                      {exec.confidence_score
                        ? `${(exec.confidence_score * 100).toFixed(0)}%`
                        : "-"}
                    </TableCell>
                    <TableCell>{formatDuration(exec.duration_ms)}</TableCell>
                    <TableCell className="text-muted-foreground">
                      {formatDate(exec.created_at)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
