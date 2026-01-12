"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import {
  getPendingReviews,
  getReviewStats,
  submitReview,
  type Review,
  type ReviewStats,
} from "@/lib/api";
import { formatDate, ConfidenceBadge } from "@/lib/formatters";

interface ReviewCardProps {
  review: Review;
  onAction: () => void;
}

function ReviewCard({ review, onAction }: ReviewCardProps) {
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [showFeedback, setShowFeedback] = useState(false);

  async function handleAction(decision: string) {
    setLoading(true);
    try {
      await submitReview(review.id, decision, feedback || undefined);
      onAction();
    } catch (err) {
      console.error("Failed to submit review:", err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg">
              Review #{review.id.slice(0, 8)}
            </CardTitle>
            <CardDescription>
              Created {formatDate(review.created_at)}
            </CardDescription>
          </div>
          <ConfidenceBadge score={review.confidence_score} />
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Original Content Preview */}
        <div>
          <h4 className="text-sm font-medium mb-2">Generated Content</h4>
          <div className="rounded-md bg-muted p-4 text-sm font-mono overflow-auto max-h-64">
            <pre className="whitespace-pre-wrap">
              {JSON.stringify(review.original_content, null, 2)}
            </pre>
          </div>
        </div>

        <Separator />

        {/* Feedback Input */}
        {showFeedback && (
          <div>
            <label className="text-sm font-medium mb-2 block">
              Feedback (optional)
            </label>
            <textarea
              className="w-full rounded-md border bg-background p-3 text-sm"
              rows={3}
              placeholder="Add feedback for improvement..."
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
            />
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          <Button
            onClick={() => handleAction("approved")}
            disabled={loading}
            className="bg-green-600 hover:bg-green-700"
          >
            Approve
          </Button>
          <Button
            variant="destructive"
            onClick={() => handleAction("rejected")}
            disabled={loading}
          >
            Reject
          </Button>
          <Button
            variant="secondary"
            onClick={() => handleAction("rerun")}
            disabled={loading}
          >
            Rerun
          </Button>
          <Button
            variant="ghost"
            onClick={() => setShowFeedback(!showFeedback)}
          >
            {showFeedback ? "Hide" : "Add"} Feedback
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default function ReviewsPage() {
  const [reviews, setReviews] = useState<Review[]>([]);
  const [stats, setStats] = useState<ReviewStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function fetchData() {
    setLoading(true);
    try {
      const [reviewsData, statsData] = await Promise.all([
        getPendingReviews(),
        getReviewStats(),
      ]);
      setReviews(reviewsData);
      setStats(statsData);
      setError(null);
    } catch (err) {
      console.error("Failed to fetch reviews:", err);
      setError("Failed to fetch reviews");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchData();
  }, []);

  if (loading && reviews.length === 0) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-48" />
        <div className="grid gap-4 md:grid-cols-4">
          <Skeleton className="h-24" />
          <Skeleton className="h-24" />
          <Skeleton className="h-24" />
          <Skeleton className="h-24" />
        </div>
        <Skeleton className="h-64" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Reviews</h1>
          <p className="text-sm text-muted-foreground">
            Human-in-the-loop review queue
          </p>
        </div>
        <Button variant="outline" onClick={fetchData} disabled={loading}>
          {loading ? "Refreshing..." : "Refresh"}
        </Button>
      </div>

      {error && (
        <div className="rounded-md bg-destructive/10 p-4 text-destructive">
          {error}
        </div>
      )}

      {/* Stats Overview */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Reviews
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.total || 0}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Pending
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-yellow-600">
              {stats?.pending || 0}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Approved
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {stats?.by_decision?.approved || 0}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Rejected
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">
              {stats?.by_decision?.rejected || 0}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Review Queue */}
      <Tabs defaultValue="pending">
        <TabsList>
          <TabsTrigger value="pending">
            Pending ({reviews.length})
          </TabsTrigger>
        </TabsList>
        <TabsContent value="pending" className="space-y-4 mt-4">
          {reviews.length === 0 ? (
            <Card>
              <CardContent className="py-8 text-center text-muted-foreground">
                No pending reviews. All caught up!
              </CardContent>
            </Card>
          ) : (
            reviews.map((review) => (
              <ReviewCard
                key={review.id}
                review={review}
                onAction={fetchData}
              />
            ))
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
