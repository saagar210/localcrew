"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { decomposeTask, researchQuery } from "@/lib/api";

interface TaskFormProps {
  onSubmit: (executionId: string) => void;
}

function DecomposeForm({ onSubmit }: TaskFormProps) {
  const [task, setTask] = useState("");
  const [project, setProject] = useState("");
  const [autoSync, setAutoSync] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!task.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const result = await decomposeTask(task, project || undefined, autoSync);
      onSubmit(result.execution_id);
      setTask("");
      setProject("");
    } catch (err) {
      setError("Failed to start decomposition. Is the server running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="text-sm font-medium mb-2 block">
          Task Description *
        </label>
        <textarea
          className="w-full rounded-md border bg-background p-3 text-sm"
          rows={4}
          placeholder="Describe the task to decompose into subtasks..."
          value={task}
          onChange={(e) => setTask(e.target.value)}
          required
        />
      </div>

      <div>
        <label className="text-sm font-medium mb-2 block">
          Project Name (optional)
        </label>
        <input
          type="text"
          className="w-full rounded-md border bg-background p-3 text-sm"
          placeholder="e.g., my-project"
          value={project}
          onChange={(e) => setProject(e.target.value)}
        />
      </div>

      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          id="autoSync"
          checked={autoSync}
          onChange={(e) => setAutoSync(e.target.checked)}
          className="rounded"
        />
        <label htmlFor="autoSync" className="text-sm">
          Auto-sync to Task Master
        </label>
      </div>

      {error && (
        <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <Button type="submit" disabled={loading || !task.trim()}>
        {loading ? "Decomposing..." : "Decompose Task"}
      </Button>
    </form>
  );
}

function ResearchForm({ onSubmit }: TaskFormProps) {
  const [query, setQuery] = useState("");
  const [depth, setDepth] = useState("medium");
  const [storeToKas, setStoreToKas] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const result = await researchQuery(query, depth, storeToKas);
      onSubmit(result.execution_id);
      setQuery("");
    } catch (err) {
      setError("Failed to start research. Is the server running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="text-sm font-medium mb-2 block">
          Research Query *
        </label>
        <textarea
          className="w-full rounded-md border bg-background p-3 text-sm"
          rows={4}
          placeholder="What would you like to research?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          required
        />
      </div>

      <div>
        <label className="text-sm font-medium mb-2 block">
          Research Depth
        </label>
        <div className="flex gap-2">
          {["shallow", "medium", "deep"].map((d) => (
            <Button
              key={d}
              type="button"
              variant={depth === d ? "default" : "outline"}
              size="sm"
              onClick={() => setDepth(d)}
            >
              {d.charAt(0).toUpperCase() + d.slice(1)}
            </Button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Shallow: Quick overview | Medium: Balanced | Deep: Comprehensive
        </p>
      </div>

      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          id="storeToKas"
          checked={storeToKas}
          onChange={(e) => setStoreToKas(e.target.checked)}
          className="rounded"
        />
        <label htmlFor="storeToKas" className="text-sm">
          Store results to Knowledge Activation System
        </label>
      </div>

      {error && (
        <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <Button type="submit" disabled={loading || !query.trim()}>
        {loading ? "Researching..." : "Start Research"}
      </Button>
    </form>
  );
}

export default function WorkflowsPage() {
  const [recentExecution, setRecentExecution] = useState<string | null>(null);

  function handleSubmit(executionId: string) {
    setRecentExecution(executionId);
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Workflows</h1>
        <p className="text-sm text-muted-foreground">
          Launch AI-powered task automation
        </p>
      </div>

      {recentExecution && (
        <Card className="border-green-200 bg-green-50">
          <CardContent className="py-4">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="bg-green-100">
                Started
              </Badge>
              <span className="text-sm">
                Execution ID: <code className="font-mono">{recentExecution}</code>
              </span>
              <a
                href="/executions"
                className="text-sm text-primary hover:underline ml-auto"
              >
                View in Executions →
              </a>
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="decompose">
        <TabsList>
          <TabsTrigger value="decompose">Task Decomposition</TabsTrigger>
          <TabsTrigger value="research">Research</TabsTrigger>
        </TabsList>

        <TabsContent value="decompose" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Task Decomposition</CardTitle>
              <CardDescription>
                Break down complex tasks into actionable subtasks using AI agents.
                Results can be automatically synced to Task Master.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <DecomposeForm onSubmit={handleSubmit} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="research" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Research Query</CardTitle>
              <CardDescription>
                Research any topic using a multi-agent pipeline. Agents will
                decompose your query, gather information, synthesize findings,
                and generate a comprehensive report.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResearchForm onSubmit={handleSubmit} />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Workflow Info */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Task Decomposition Crew</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground space-y-2">
            <p><strong>Analyzer Agent:</strong> Understands task context and requirements</p>
            <p><strong>Planner Agent:</strong> Creates structured subtask breakdown</p>
            <p><strong>Validator Agent:</strong> Ensures completeness and coherence</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Research Crew</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground space-y-2">
            <p><strong>Query Decomposer:</strong> Breaks query into sub-questions</p>
            <p><strong>Gatherer:</strong> Searches and evaluates sources</p>
            <p><strong>Synthesizer:</strong> Combines findings with confidence scores</p>
            <p><strong>Reporter:</strong> Generates formatted report with citations</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
