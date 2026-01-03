"""
Subscription Tiers and Billing

This module defines subscription plans, their limits, and billing logic.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from enum import Enum


class SubscriptionTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class TierLimits:
    """Limits for a subscription tier"""
    max_documents: int
    max_document_size_mb: int
    max_total_storage_mb: int
    max_meeting_minutes_per_day: int
    max_questions_per_day: int
    max_answers_per_day: int
    features: list[str]


# Define subscription tiers
SUBSCRIPTION_TIERS = {
    SubscriptionTier.FREE: TierLimits(
        max_documents=5,
        max_document_size_mb=5,
        max_total_storage_mb=50,
        max_meeting_minutes_per_day=30,
        max_questions_per_day=50,
        max_answers_per_day=20,
        features=[
            "basic_transcription",
            "question_detection",
            "basic_answers",
        ]
    ),
    SubscriptionTier.PRO: TierLimits(
        max_documents=50,
        max_document_size_mb=25,
        max_total_storage_mb=500,
        max_meeting_minutes_per_day=240,
        max_questions_per_day=500,
        max_answers_per_day=200,
        features=[
            "basic_transcription",
            "question_detection",
            "basic_answers",
            "document_search",
            "ai_suggestions",
            "meeting_history",
            "export_transcripts",
        ]
    ),
    SubscriptionTier.ENTERPRISE: TierLimits(
        max_documents=500,
        max_document_size_mb=100,
        max_total_storage_mb=5000,
        max_meeting_minutes_per_day=1440,  # 24 hours
        max_questions_per_day=10000,
        max_answers_per_day=5000,
        features=[
            "basic_transcription",
            "question_detection",
            "basic_answers",
            "document_search",
            "ai_suggestions",
            "meeting_history",
            "export_transcripts",
            "custom_api_keys",
            "priority_support",
            "team_management",
            "analytics",
        ]
    ),
}


def get_tier_limits(tier: str) -> TierLimits:
    """Get limits for a subscription tier"""
    try:
        tier_enum = SubscriptionTier(tier.lower())
        return SUBSCRIPTION_TIERS[tier_enum]
    except (ValueError, KeyError):
        return SUBSCRIPTION_TIERS[SubscriptionTier.FREE]


def has_feature(tier: str, feature: str) -> bool:
    """Check if a tier has a specific feature"""
    limits = get_tier_limits(tier)
    return feature in limits.features


class UsageLimitError(Exception):
    """Exception raised when a usage limit is exceeded"""
    def __init__(self, limit_type: str, limit_value: int, current_value: int):
        self.limit_type = limit_type
        self.limit_value = limit_value
        self.current_value = current_value
        super().__init__(
            f"Usage limit exceeded: {limit_type} "
            f"(limit: {limit_value}, current: {current_value})"
        )


class SubscriptionManager:
    """
    Manages user subscriptions and usage limits.

    Tracks usage and enforces limits based on subscription tier.
    """

    def __init__(self):
        # In-memory usage tracking (resets daily)
        # In production, this should be persisted to the database
        self._daily_usage: dict[str, dict[str, int]] = {}
        self._usage_reset_date: dict[str, str] = {}

    def _get_today(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%d")

    def _reset_if_needed(self, user_id: str):
        """Reset daily usage if it's a new day"""
        today = self._get_today()
        if self._usage_reset_date.get(user_id) != today:
            self._daily_usage[user_id] = {
                "meeting_minutes": 0,
                "questions": 0,
                "answers": 0,
            }
            self._usage_reset_date[user_id] = today

    def _get_user_usage(self, user_id: str) -> dict[str, int]:
        """Get user's daily usage"""
        self._reset_if_needed(user_id)
        return self._daily_usage.get(user_id, {
            "meeting_minutes": 0,
            "questions": 0,
            "answers": 0,
        })

    def check_meeting_limit(self, user_id: str, tier: str, minutes: int = 1) -> bool:
        """Check if user can use more meeting minutes"""
        limits = get_tier_limits(tier)
        usage = self._get_user_usage(user_id)
        return usage.get("meeting_minutes", 0) + minutes <= limits.max_meeting_minutes_per_day

    def check_question_limit(self, user_id: str, tier: str) -> bool:
        """Check if user can ask more questions"""
        limits = get_tier_limits(tier)
        usage = self._get_user_usage(user_id)
        return usage.get("questions", 0) < limits.max_questions_per_day

    def check_answer_limit(self, user_id: str, tier: str) -> bool:
        """Check if user can get more answers"""
        limits = get_tier_limits(tier)
        usage = self._get_user_usage(user_id)
        return usage.get("answers", 0) < limits.max_answers_per_day

    def check_document_limit(self, user_id: str, tier: str, current_count: int) -> bool:
        """Check if user can upload more documents"""
        limits = get_tier_limits(tier)
        return current_count < limits.max_documents

    def check_storage_limit(self, user_id: str, tier: str, current_mb: float, new_file_mb: float) -> bool:
        """Check if user has storage capacity"""
        limits = get_tier_limits(tier)
        return (current_mb + new_file_mb) <= limits.max_total_storage_mb

    def check_file_size_limit(self, tier: str, file_size_mb: float) -> bool:
        """Check if file size is within limits"""
        limits = get_tier_limits(tier)
        return file_size_mb <= limits.max_document_size_mb

    def record_meeting_minutes(self, user_id: str, minutes: int = 1):
        """Record meeting minutes used"""
        self._reset_if_needed(user_id)
        if user_id not in self._daily_usage:
            self._daily_usage[user_id] = {"meeting_minutes": 0, "questions": 0, "answers": 0}
        self._daily_usage[user_id]["meeting_minutes"] += minutes

    def record_question(self, user_id: str):
        """Record a question asked"""
        self._reset_if_needed(user_id)
        if user_id not in self._daily_usage:
            self._daily_usage[user_id] = {"meeting_minutes": 0, "questions": 0, "answers": 0}
        self._daily_usage[user_id]["questions"] += 1

    def record_answer(self, user_id: str):
        """Record an answer provided"""
        self._reset_if_needed(user_id)
        if user_id not in self._daily_usage:
            self._daily_usage[user_id] = {"meeting_minutes": 0, "questions": 0, "answers": 0}
        self._daily_usage[user_id]["answers"] += 1

    def get_usage_summary(self, user_id: str, tier: str) -> dict:
        """Get user's usage summary with limits"""
        limits = get_tier_limits(tier)
        usage = self._get_user_usage(user_id)

        return {
            "tier": tier,
            "usage": {
                "meeting_minutes": {
                    "used": usage.get("meeting_minutes", 0),
                    "limit": limits.max_meeting_minutes_per_day,
                    "remaining": max(0, limits.max_meeting_minutes_per_day - usage.get("meeting_minutes", 0))
                },
                "questions": {
                    "used": usage.get("questions", 0),
                    "limit": limits.max_questions_per_day,
                    "remaining": max(0, limits.max_questions_per_day - usage.get("questions", 0))
                },
                "answers": {
                    "used": usage.get("answers", 0),
                    "limit": limits.max_answers_per_day,
                    "remaining": max(0, limits.max_answers_per_day - usage.get("answers", 0))
                },
            },
            "limits": {
                "max_documents": limits.max_documents,
                "max_document_size_mb": limits.max_document_size_mb,
                "max_total_storage_mb": limits.max_total_storage_mb,
            },
            "features": limits.features,
            "resets_at": f"{self._get_today()}T00:00:00Z"
        }


# Global subscription manager
subscription_manager = SubscriptionManager()
