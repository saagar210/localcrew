"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { getExecutions, type Execution } from "@/lib/api";
import { StatusBadge, formatDuration, formatDate } from "@/lib/formatters";

export default function ExecutionsPage() {
  const [executions, setExecutions] = useState<Execution[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [limit, setLimit] = useState(25);

  const fetchExecutions = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getExecutions(limit);
      setExecutions(data);
      setError(null);
    } catch (err) {
      console.error("Failed to fetch executions:", err);
      setError("Failed to fetch executions");
    } finally {
      setLoading(false);
    }
  }, [limit]);

  useEffect(() => {
    fetchExecutions();
  }, [fetchExecutions]);

  if (loading && executions.length === 0) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-96" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Executions</h1>
          <p className="text-sm text-muted-foreground">
            History of all crew executions
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={fetchExecutions} disabled={loading}>
            {loading ? "Refreshing..." : "Refresh"}
          </Button>
        </div>
      </div>

      {error && (
        <div className="rounded-md bg-destructive/10 p-4 text-destructive">
          {error}
        </div>
      )}

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>All Executions</CardTitle>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-muted-foreground">Show:</span>
              {[25, 50, 100].map((n) => (
                <Button
                  key={n}
                  variant={limit === n ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setLimit(n)}
                >
                  {n}
                </Button>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {executions.length === 0 ? (
            <p className="text-center text-muted-foreground py-8">
              No executions found.
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[100px]">ID</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Input</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Duration</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead>Completed</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {executions.map((exec) => (
                  <TableRow key={exec.id}>
                    <TableCell className="font-mono text-xs">
                      {exec.id.slice(0, 8)}...
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline">{exec.crew_type}</Badge>
                    </TableCell>
                    <TableCell className="max-w-[250px] truncate" title={exec.input_text}>
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
                    <TableCell className="text-muted-foreground text-sm">
                      {formatDate(exec.created_at)}
                    </TableCell>
                    <TableCell className="text-muted-foreground text-sm">
                      {exec.completed_at ? formatDate(exec.completed_at) : "-"}
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
