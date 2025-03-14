import json
import logging
import os
import sys
import time
import traceback
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict

import pandas as pd
from bson import ObjectId
from bson.json_util import dumps, loads
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from numpy import int8
from openai import OpenAI
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to MongoDB
uri = "mongodb+srv://internlink61:feH2kOcVJHGqVvyG@internlink.trwsw.mongodb.net/database?retryWrites=true&w=majority&appName=Internlink"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi("1"))
db = client["database"]
applicants_collection = db["applicants"]


app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

# Constants
TAGS = [
    "IT",
    "Marketing",
    "Finance and Accounting",
    "Sales",
    "Human Resources",
    "Legal",
    "Retail",
    "Customer Service",
]
LANGUAGES = [
    "english",
    "hindi",
    "marathi",
    "gujarati",
    "spanish",
    "french",
    "urdu",
    "kannada",
    "tamil",
]
JOB_TYPES = ["internship", "jobs"]
LOCATIONS = ["delhi", "mumbai", "bangalore", "hyderabad", "pune", "chennai"]


def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle and format exceptions"""
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    return {
        "type": exc_type.__name__,
        "message": str(exc_value),
        "traceback": "".join(traceback.format_tb(exc_traceback)),
    }


def transform_mongodb_to_features(applicant):
    # Current version only handles basic profile info
    # Need to add new fields
    features = {
        "uid": str(applicant["_id"]),
        "name": applicant["name"],
        "age": applicant["profile"].get("age", 0),
        "experience": len(applicant["profile"].get("experience", [])),
        "weekly_hours": 40,
        "profile": applicant["profile"],
        # Add new features
        "personality_score": calculate_personality_score(
            applicant["profile"].get("personalityBlueprint", [])
        ),
        "education_level": get_education_level(
            applicant["profile"].get("education", {})
        ),
        "total_applications": len(applicant.get("appliedJobs", [])),
        "is_resume_parsed": applicant.get("isResumeParsed", False),
    }

    # Add language features
    for lang in LANGUAGES:
        features[lang] = int(
            any(
                l["language"].lower() == lang.lower()
                for l in applicant["profile"].get("languages", [])
            )
        )

    # Add tags features
    for tag in TAGS:
        features[tag] = int(tag in applicant["profile"].get("tags", []))

    features["seeking"] = applicant["profile"].get("seeking", "jobs")

    return features


def mongo_serialize(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize MongoDB objects to JSON-compatible format"""
    try:
        # Convert to BSON and back to handle MongoDB-specific types
        serialized = loads(dumps(obj))

        # Additional cleaning for interpretation
        def clean_value(v):
            if isinstance(v, (ObjectId, datetime, date)):
                return str(v)
            if isinstance(v, dict):
                return {k: clean_value(val) for k, val in v.items()}
            if isinstance(v, list):
                return [clean_value(item) for item in v]
            return v

        return clean_value(serialized)

    except Exception as e:
        logger.error(f"Serialization error: {str(e)}")
        return str(obj)


def calculate_personality_score(blueprint):
    if not blueprint:
        return 0
    # Calculate a normalized score based on personality answers
    # Assuming max option is 4
    return sum(answer["selectedOption"] for answer in blueprint) / (len(blueprint) * 4)


def get_education_level(education):
    if not education:
        return 0
    # Map education year to a numerical level
    return education.get("year", 0)


def calculate_personality_match(blueprint1, blueprint2):
    if not blueprint1 or not blueprint2:
        return 0

    matching_answers = sum(
        1
        for a1, a2 in zip(blueprint1, blueprint2)
        if a1["questionId"] == a2["questionId"]
        and a1["selectedOption"] == a2["selectedOption"]
    )
    return (matching_answers / len(blueprint1)) * 100 if blueprint1 else 0


def get_communication_score(applicant):
    """Extract and normalize communication score from an applicant's call statistics."""
    try:
        # Get the most recent call data (if available)
        hollr_ai_calls = applicant.get("profile", {}).get("hollrAiCalls", [])

        if not hollr_ai_calls or len(hollr_ai_calls) == 0:
            return None

        # Retrieve the most recent call
        latest_call = None

        # Check if we have direct data or references
        if (
            isinstance(hollr_ai_calls[0], dict)
            and "confidenceScore" in hollr_ai_calls[0]
        ):
            # Direct data available
            latest_call = max(hollr_ai_calls, key=lambda x: x.get("createdAt", 0))
        else:
            # We have references - need to fetch the actual data
            call_ids = [call_id for call_id in hollr_ai_calls if call_id]
            if not call_ids:
                return None

            # Get the latest call by querying the database
            latest_call_id = call_ids[-1]  # Assuming newest is last in the array
            latest_call = db["HollrAiCall"].find_one({"_id": latest_call_id})

        if not latest_call:
            return None

        # Calculate weighted communication score from metrics
        metric_weights = {
            "userClarity": 0.4,  # Clear articulation is most important
            "userAccuracy": 0.35,  # Understanding context is very important
            "confidenceScore": 0.25,  # Confidence is important but not as critical
        }

        communication_score = 0
        metrics_present = False

        for metric, weight in metric_weights.items():
            if metric in latest_call and latest_call[metric] is not None:
                # Normalize to 0-1 scale
                normalized_value = latest_call[metric] / 100  # Assuming 0-100 scale
                normalized_value = max(0, min(1, normalized_value))  # Ensure in range
                communication_score += weight * normalized_value
                metrics_present = True

        return communication_score if metrics_present else None

    except Exception as e:
        logger.error(f"Error calculating communication score: {str(e)}")
        return None


def adjust_similarity_with_communication(similarity_score, applicant):
    """Adjust the similarity score based on the applicant's communication abilities."""
    # Get communication score for the real applicant
    comm_score = get_communication_score(applicant)

    # If no communication data is available, return the original score
    if comm_score is None:
        return similarity_score

    # Apply a more conservative adjustment factor
    # - Instead of ±20%, use ±10% to avoid overriding other factors
    # - This creates a 20% potential adjustment range centered around 1.0
    adjustment_factor = 0.9 + (comm_score * 0.2)

    # Apply the adjustment to the similarity score
    adjusted_score = similarity_score * adjustment_factor

    # Ensure the score stays within valid range (0-100)
    return round(max(0, min(100, adjusted_score)), 2)


def calculate_similarity(applicant1, applicant2):
    """Calculate similarity between two applicants with weighted features"""

    # Normalize age (assuming typical working age range 18-65)
    def normalize_age(age):
        return (age - 18) / (65 - 18)

    # Normalize experience (assuming max 40 years)
    def normalize_experience(exp):
        return exp / 40

    # Weights for different feature types
    weights = {
        "age": 0.10,
        "experience": 0.15,
        "languages": 0.15,
        "tags": 0.15,
        "skills": 0.20,
        "education": 0.15,
        "personality": 0.10,  # New weight for personality matching
    }

    # Calculate weighted components
    age_sim = 1 - abs(
        normalize_age(applicant1["age"]) - normalize_age(applicant2["age"])
    )
    exp_sim = 1 - abs(
        normalize_experience(applicant1["experience"])
        - normalize_experience(applicant2["experience"])
    )

    # Language similarity (Jaccard similarity)
    lang1 = set(i for i in LANGUAGES if applicant1[i])
    lang2 = set(i for i in LANGUAGES if applicant2[i])
    lang_sim = len(lang1.intersection(lang2)) / max(len(lang1.union(lang2)), 1)

    # Tags similarity (Jaccard similarity)
    tags1 = set(t for t in TAGS if applicant1[t])
    tags2 = set(t for t in TAGS if applicant2[t])
    tags_sim = len(tags1.intersection(tags2)) / max(len(tags1.union(tags2)), 1)

    # Skills similarity (need to add to transform_mongodb_to_features first)
    skills1 = set(
        s["skillName"].lower() for s in applicant1.get("profile", {}).get("skills", [])
    )
    skills2 = set(
        s["skillName"].lower() for s in applicant2.get("profile", {}).get("skills", [])
    )
    skills_sim = len(skills1.intersection(skills2)) / max(
        len(skills1.union(skills2)), 1
    )

    # Education similarity
    edu_sim = 0.0
    edu1 = applicant1.get("profile", {}).get("education", {})
    edu2 = applicant2.get("profile", {}).get("education", {})

    # Compare different aspects of education
    if edu1 and edu2:
        same_major = int(edu1.get("major", "").lower() == edu2.get("major", "").lower())
        same_level = int(edu1.get("institutionType") == edu2.get("institutionType"))
        cgpa_diff = abs(float(edu1.get("cgpa", 0)) - float(edu2.get("cgpa", 0))) / 10.0

        edu_sim = (same_major + same_level + (1 - cgpa_diff)) / 3

    personality_sim = 1 - abs(
        applicant1.get("personality_score", 0) - applicant2.get("personality_score", 0)
    )

    total_similarity = (
        weights["age"] * age_sim
        + weights["experience"] * exp_sim
        + weights["languages"] * lang_sim
        + weights["tags"] * tags_sim
        + weights["skills"] * skills_sim
        + weights["education"] * edu_sim
        + weights["personality"] * personality_sim
    )

    return round(total_similarity * 100, 2)


def find_similar_applicants(target_applicant, k=5):
    """Find k most similar applicants with detailed filtering and matching"""
    all_applicants = list(applicants_collection.find({}))
    feature_vectors = [transform_mongodb_to_features(app) for app in all_applicants]

    # Initial filtering
    filtered_applicants = []
    for app in feature_vectors:
        # Basic filters
        same_seeking = app["seeking"] == target_applicant["seeking"]

        # Age range filter (within 5 years)
        age_diff = abs(app["age"] - target_applicant["age"])
        age_compatible = age_diff <= 5

        # Experience level compatibility
        exp_diff = abs(app["experience"] - target_applicant["experience"])
        exp_compatible = exp_diff <= 2  # Within 2 years of experience difference

        # Education level compatibility
        target_edu = target_applicant.get("profile", {}).get("education", {})
        app_edu = app["profile"].get("education", {})
        edu_compatible = target_edu.get("institutionType") == app_edu.get(
            "institutionType"
        )

        # Skills overlap check
        target_skills = set(
            s["skillName"].lower()
            for s in target_applicant.get("profile", {}).get("skills", [])
        )
        app_skills = set(
            s["skillName"].lower() for s in app["profile"].get("skills", [])
        )
        skills_overlap = len(target_skills.intersection(app_skills)) > 0

        # Apply all filters
        if (
            same_seeking
            and age_compatible
            and exp_compatible
            and edu_compatible
            and skills_overlap
        ):
            filtered_applicants.append(app)

    # Calculate detailed similarities for filtered candidates
    similarities = []
    for app in filtered_applicants:
        if app["uid"] != target_applicant.get("uid"):  # Exclude self-comparison
            similarity_score = calculate_similarity(target_applicant, app)

            # Calculate specific match factors
            match_factors = {
                "skills_match": len(
                    set(
                        s["skillName"].lower() for s in app["profile"].get("skills", [])
                    ).intersection(
                        set(
                            s["skillName"].lower()
                            for s in target_applicant.get("profile", {}).get(
                                "skills", []
                            )
                        )
                    )
                ),
                "language_match": len(
                    set(
                        l["language"].lower()
                        for l in app["profile"].get("languages", [])
                    ).intersection(
                        set(
                            l["language"].lower()
                            for l in target_applicant.get("profile", {}).get(
                                "languages", []
                            )
                        )
                    )
                ),
                "tags_match": len(
                    set(app["profile"].get("tags", [])).intersection(
                        set(target_applicant.get("profile", {}).get("tags", []))
                    )
                ),
                "education_level_match": (
                    app["profile"].get("education", {}).get("institutionType")
                    == target_applicant.get("profile", {})
                    .get("education", {})
                    .get("institutionType")
                ),
                "experience_gap": abs(
                    app["experience"] - target_applicant["experience"]
                ),
            }

            similarities.append(
                {
                    "applicant": app,
                    "similarity_score": similarity_score,
                    "match_factors": match_factors,
                }
            )

    # Sort by similarity score and get top k
    sorted_similarities = sorted(
        similarities,
        key=lambda x: (x["similarity_score"], x["match_factors"]["skills_match"]),
        reverse=True,
    )[:k]

    # Format the results
    results = []
    for item in sorted_similarities:
        app = item["applicant"]
        results.append(
            {
                "id": app["uid"],
                "name": app["name"],
                "similarity_score": item["similarity_score"],
                "match_factors": item["match_factors"],
                "profile": {
                    "age": app["age"],
                    "experience": app["experience"],
                    "education": app["profile"].get("education"),
                    "skills": app["profile"].get("skills"),
                    "languages": app["profile"].get("languages"),
                    "tags": app["profile"].get("tags"),
                    "seeking": app["seeking"],
                },
            }
        )

    return results


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify MongoDB connection and service status.
    """
    try:
        # Test MongoDB connection by fetching a single document
        applicants_collection.find_one()
        return (
            jsonify(
                {
                    "status": "healthy",
                    "details": {
                        "mongo_db_connection": "successful",
                        "service_status": "operational",
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            200,
        )
    except Exception as e:
        # Log the error for debugging purposes
        app.logger.error(f"Health check failed: {str(e)}")
        return (
            jsonify(
                {
                    "status": "error",
                    "details": {
                        "mongo_db_connection": "failed",
                        "service_status": "unavailable",
                    },
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            500,
        )


def serialize_mongo_doc(doc):
    """Convert MongoDB document to JSON-serializable format."""
    if isinstance(doc, dict):
        return {k: serialize_mongo_doc(v) for k, v in doc.items()}
    elif isinstance(doc, list):
        return [serialize_mongo_doc(v) for v in doc]
    elif isinstance(doc, ObjectId):
        return str(doc)
    elif isinstance(doc, bytes):
        try:
            return doc.decode("utf-8")
        except UnicodeDecodeError:
            return str(doc)
    elif isinstance(doc, (datetime, date)):
        return doc.isoformat()
    elif isinstance(doc, Decimal):
        return str(doc)
    elif isinstance(doc, (int, float, str, bool, None.__class__)):
        return doc
    else:
        try:
            # Try to convert to string as last resort
            return str(doc)
        except Exception as e:
            logger.error(f"Cannot serialize object of type {type(doc)}: {str(e)}")
            return None


@app.route("/api/applicants/all", methods=["GET"])
@cross_origin()
def get_all_applicants():
    """Get all applicants from the database."""
    try:
        # Get all applicants from database
        applicants = list(applicants_collection.find({}))

        # Convert applicants to JSON-serializable format
        serialized_applicants = []
        for applicant in applicants:
            try:
                # Convert the document to JSON-serializable format
                serialized_doc = serialize_mongo_doc(applicant)

                # Remove sensitive information
                serialized_doc.pop("password", None)
                serialized_doc.pop("otp", None)
                serialized_doc.pop("otpExpiry", None)

                serialized_applicants.append(serialized_doc)
            except Exception as e:
                logger.error(f"Error serializing document: {str(e)}")
                logger.error(f"Problematic document: {applicant}")
                continue  # Skip problematic documents

        return jsonify(serialized_applicants), 200

    except Exception as e:
        error_info = handle_exception(*sys.exc_info())
        return jsonify({"error": error_info}), 500


@app.route("/api/applicants/compare", methods=["POST"])
@cross_origin()
def compare_applicants():
    """
    Compare two applicants and provide detailed similarity analysis.

    Input JSON Format:
    {
        "applicant1_id": "507f1f77bcf86cd799439011",  # MongoDB ObjectId as string
        "applicant2_id": "507f1f77bcf86cd799439012"   # MongoDB ObjectId as string
    }

    Returns:
    {
        "similarity_score": 85.5,  # Overall similarity percentage
        "match_factors": {
            "skills_match": {
                "common": ["Python", "Java"],
                "only_applicant1": ["React"],
                "only_applicant2": ["Angular"]
            },
            "education": {
                "same_level": true,
                "same_major": false,
                "cgpa_difference": 0.5
            },
            "experience_gap": 1
        },
        "applicant1_details": {
            "name": "John Doe",
            "profile": {
                "age": 25,
                "education": {
                    "institutionName": "MIT",
                    "institutionType": "college",
                    "major": "Computer Science",
                    "cgpa": "3.8"
                },
                "experience": [...],
                "skills": [...],
                "languages": [...],
                "tags": [...]
            }
        },
        "applicant2_details": {
            "name": "Jane Smith",
            "profile": {
                // Similar structure as applicant1_details
            }
        }
    }

    Error Response:
    {
        "error": {
            "type": "ValueError",
            "message": "Invalid applicant ID",
            "traceback": "..."
        }
    }
    """
    try:
        data = request.get_json()
        applicant1_id = ObjectId(data.get("applicant1_id"))
        applicant2_id = ObjectId(data.get("applicant2_id"))

        applicant1 = applicants_collection.find_one({"_id": applicant1_id})
        applicant2 = applicants_collection.find_one({"_id": applicant2_id})

        if not applicant1 or not applicant2:
            return jsonify({"error": "Applicant not found"}), 404

        feat1 = transform_mongodb_to_features(applicant1)
        feat2 = transform_mongodb_to_features(applicant2)

        similarity_score = calculate_similarity(feat1, feat2)

        # Calculate match factors
        match_factors = {
            "skills_match": {
                "common": list(
                    set(
                        s["skillName"].lower()
                        for s in applicant1["profile"].get("skills", [])
                    ).intersection(
                        set(
                            s["skillName"].lower()
                            for s in applicant2["profile"].get("skills", [])
                        )
                    )
                ),
                "only_applicant1": list(
                    set(
                        s["skillName"].lower()
                        for s in applicant1["profile"].get("skills", [])
                    )
                    - set(
                        s["skillName"].lower()
                        for s in applicant2["profile"].get("skills", [])
                    )
                ),
                "only_applicant2": list(
                    set(
                        s["skillName"].lower()
                        for s in applicant2["profile"].get("skills", [])
                    )
                    - set(
                        s["skillName"].lower()
                        for s in applicant1["profile"].get("skills", [])
                    )
                ),
            },
            "education": {
                "same_level": applicant1["profile"]
                .get("education", {})
                .get("institutionType")
                == applicant2["profile"].get("education", {}).get("institutionType"),
                "same_major": applicant1["profile"].get("education", {}).get("major")
                == applicant2["profile"].get("education", {}).get("major"),
                "cgpa_difference": abs(
                    float(applicant1["profile"].get("education", {}).get("cgpa", 0))
                    - float(applicant2["profile"].get("education", {}).get("cgpa", 0))
                ),
            },
            "experience_gap": abs(
                len(applicant1["profile"].get("experience", []))
                - len(applicant2["profile"].get("experience", []))
            ),
        }

        response = {
            "similarity_score": similarity_score,
            "match_factors": match_factors,
            "applicant1_details": {
                "name": applicant1["name"],
                "profile": {
                    "age": applicant1["profile"].get("age"),
                    "education": applicant1["profile"].get("education"),
                    "experience": applicant1["profile"].get("experience"),
                    "skills": applicant1["profile"].get("skills"),
                    "languages": applicant1["profile"].get("languages"),
                    "tags": applicant1["profile"].get("tags"),
                },
            },
            "applicant2_details": {
                "name": applicant2["name"],
                "profile": {
                    "age": applicant2["profile"].get("age"),
                    "education": applicant2["profile"].get("education"),
                    "experience": applicant2["profile"].get("experience"),
                    "skills": applicant2["profile"].get("skills"),
                    "languages": applicant2["profile"].get("languages"),
                    "tags": applicant2["profile"].get("tags"),
                },
            },
        }

        return jsonify(response), 200

    except Exception as e:
        error_info = handle_exception(*sys.exc_info())
        return jsonify({"error": error_info}), 500


@app.route("/api/applicants/compare-profile", methods=["POST"])
@cross_origin()
def compare_profile_with_applicant():
    """
    Compare a complete profile with an existing applicant.

    Input JSON Format:
    {
        "applicant_id": "507f1f77bcf86cd799439011",
        "profile": {
            "name": "Test User",
            "age": 25,
            "experience": [
                {
                    "company": "Tech Corp",
                    "position": "Software Engineer",
                    "startYear": "2020-01-01",
                    "endYear": "2022-12-31"
                }
            ],
            "education": {
                "institutionName": "MIT",
                "institutionType": "college",
                "major": "Computer Science",
                "cgpa": "3.8"
            },
            "skills": [
                {"skillName": "Python", "proficiency": "Advanced"},
                {"skillName": "Java", "proficiency": "Intermediate"}
            ],
            "languages": [
                {"language": "English", "proficiency": "Native"},
                {"language": "Spanish", "proficiency": "Intermediate"}
            ],
            "tags": ["IT", "Marketing"],
            "seeking": "jobs"
        }
    }

    Returns:
    {
        "similarity_score": 85.5,
        "match_factors": {
            "skills_match": {
                "common": ["Python", "Java"],
                "only_profile": ["React"],
                "only_applicant": ["Angular"]
            },
            "education": {
                "same_level": true,
                "same_major": false,
                "cgpa_difference": 0.5
            },
            "experience_gap": 1,
            "language_match": ["English"],
            "tags_match": ["IT"]
        },
        "applicant_details": {
            "name": "John Doe",
            "profile": {
                "age": 25,
                "education": {...},
                "experience": [...],
                "skills": [...],
                "languages": [...],
                "tags": [...]
            }
        }
    }

    Error Response:
    {
        "error": {
            "type": "ValueError",
            "message": "Invalid input format",
            "traceback": "..."
        }
    }
    """
    try:
        data = request.get_json()
        applicant_id = data.get("applicant_id")
        profile = data.get("profile", {})

        # Validate input
        if not applicant_id or not profile:
            return jsonify({"error": "Missing required fields"}), 400

        # Get existing applicant
        applicant = applicants_collection.find_one({"_id": ObjectId(applicant_id)})
        if not applicant:
            return jsonify({"error": "Applicant not found"}), 404

        # Transform input profile to feature format
        profile_features = {
            "uid": "temp",
            "name": profile.get("name", "Temporary"),
            "age": profile.get("age", 0),
            "experience": len(profile.get("experience", [])),
            "weekly_hours": profile.get("weekly_hours", 40),
            "profile": profile,
        }

        # Add language features
        for lang in LANGUAGES:
            profile_features[lang] = int(
                lang.lower()
                in [l["language"].lower() for l in profile.get("languages", [])]
            )

        # Add tags features
        for tag in TAGS:
            profile_features[tag] = int(tag in profile.get("tags", []))

        profile_features["seeking"] = profile.get("seeking", "jobs")

        # Transform existing applicant to feature format
        applicant_features = transform_mongodb_to_features(applicant)

        # Calculate similarity score
        similarity_score = calculate_similarity(profile_features, applicant_features)

        # Calculate detailed match factors
        match_factors = {
            "skills_match": {
                "common": list(
                    set(
                        s["skillName"].lower() for s in profile.get("skills", [])
                    ).intersection(
                        set(
                            s["skillName"].lower()
                            for s in applicant["profile"].get("skills", [])
                        )
                    )
                ),
                "only_profile": list(
                    set(s["skillName"].lower() for s in profile.get("skills", []))
                    - set(
                        s["skillName"].lower()
                        for s in applicant["profile"].get("skills", [])
                    )
                ),
                "only_applicant": list(
                    set(
                        s["skillName"].lower()
                        for s in applicant["profile"].get("skills", [])
                    )
                    - set(s["skillName"].lower() for s in profile.get("skills", []))
                ),
            },
            "education": {
                "same_level": profile.get("education", {}).get("institutionType")
                == applicant["profile"].get("education", {}).get("institutionType"),
                "same_major": profile.get("education", {}).get("major")
                == applicant["profile"].get("education", {}).get("major"),
                "cgpa_difference": abs(
                    float(profile.get("education", {}).get("cgpa", 0))
                    - float(applicant["profile"].get("education", {}).get("cgpa", 0))
                ),
            },
            "experience_gap": abs(
                len(profile.get("experience", []))
                - len(applicant["profile"].get("experience", []))
            ),
            "language_match": list(
                set(
                    l["language"].lower() for l in profile.get("languages", [])
                ).intersection(
                    set(
                        l["language"].lower()
                        for l in applicant["profile"].get("languages", [])
                    )
                )
            ),
            "tags_match": list(
                set(profile.get("tags", [])).intersection(
                    set(applicant["profile"].get("tags", []))
                )
            ),
        }

        response = {
            "similarity_score": similarity_score,
            "match_factors": match_factors,
            "applicant_details": {
                "name": applicant["name"],
                "profile": {
                    "age": applicant["profile"].get("age"),
                    "education": applicant["profile"].get("education"),
                    "experience": applicant["profile"].get("experience"),
                    "skills": applicant["profile"].get("skills"),
                    "languages": applicant["profile"].get("languages"),
                    "tags": applicant["profile"].get("tags"),
                },
            },
        }

        response = mongo_serialize(response)

        return jsonify(response), 200

    except Exception as e:
        error_info = handle_exception(*sys.exc_info())
        return jsonify({"error": error_info}), 500


@app.route("/api/applicants/recommend", methods=["POST"])
@cross_origin()
def recommend_applicants():
    """
    Get applicant recommendations based on specified requirements.

    Input JSON Format:
    {
        "requirements": {
            "min_age": 20,
            "max_age": 30,
            "languages": ["English", "Hindi"],
            "min_experience": 2,
            "required_skills": ["Python", "Java", "React"],
            "tags": ["IT", "Marketing"],
            "education_level": "college",
            "major": "Computer Science"
        }
    }

    Returns:
    [
        {
            "id": "507f1f77bcf86cd799439011",
            "name": "John Doe",
            "match_score": 92.5,
            "profile": {
                "age": 25,
                "experience": 3,
                "education": {
                    "institutionName": "MIT",
                    "institutionType": "college",
                    "major": "Computer Science",
                    "cgpa": "3.8"
                },
                "languages": ["English", "Hindi"],
                "skills": ["Python", "Java", "React"],
                "tags": ["IT", "Marketing"]
            }
        },
        // ... more recommendations
    ]

    Error Response:
    {
        "error": {
            "type": "ValueError",
            "message": "Invalid requirement format",
            "traceback": "..."
        }
    }
    """
    try:
        data = request.get_json()
        requirements = data.get("requirements", {})

        # Build MongoDB query based on requirements
        query = {}

        # Age requirement
        if "min_age" in requirements:
            query["profile.age"] = {"$gte": requirements["min_age"]}
        if "max_age" in requirements:
            query.setdefault("profile.age", {})["$lte"] = requirements["max_age"]

        # Language requirements
        if "languages" in requirements:
            query["profile.languages.language"] = {
                "$in": [lang.lower() for lang in requirements["languages"]]
            }

        # Experience requirement
        if "min_experience" in requirements:
            query["$expr"] = {
                "$gte": [
                    {"$size": "$profile.experience"},
                    requirements["min_experience"],
                ]
            }

        # Skills requirement
        if "required_skills" in requirements:
            query["profile.skills.skillName"] = {"$in": requirements["required_skills"]}

        # Tags requirement
        if "tags" in requirements:
            query["profile.tags"] = {"$in": requirements["tags"]}

        # Education requirements
        if "education_level" in requirements:
            query["profile.education.institutionType"] = requirements["education_level"]

        if "major" in requirements:
            query["profile.education.major"] = requirements["major"]

        # Get matching applicants
        matching_applicants = list(applicants_collection.find(query))

        # Transform to feature vectors and calculate similarities
        if matching_applicants:
            feature_vectors = [
                transform_mongodb_to_features(app) for app in matching_applicants
            ]

            recommendations = []
            for i, app in enumerate(matching_applicants):
                # Create a mock target profile from requirements
                target_features = {
                    "age": requirements.get("min_age", 0),
                    "experience": requirements.get("min_experience", 0),
                    "profile": {
                        "skills": [
                            {"skillName": skill}
                            for skill in requirements.get("required_skills", [])
                        ],
                        "languages": [
                            {"language": lang}
                            for lang in requirements.get("languages", [])
                        ],
                        "tags": requirements.get("tags", []),
                        "education": {
                            "institutionType": requirements.get("education_level", ""),
                            "major": requirements.get("major", ""),
                        },
                    },
                }

                similarity_score = calculate_similarity(
                    feature_vectors[i], target_features
                )

                recommendations.append(
                    {
                        "id": str(app["_id"]),
                        "name": app["name"],
                        "match_score": similarity_score,
                        "profile": {
                            "age": app["profile"].get("age"),
                            "experience": len(app["profile"].get("experience", [])),
                            "education": app["profile"].get("education"),
                            "languages": [
                                l["language"]
                                for l in app["profile"].get("languages", [])
                            ],
                            "skills": [
                                s["skillName"] for s in app["profile"].get("skills", [])
                            ],
                            "tags": app["profile"].get("tags", []),
                        },
                    }
                )

            # Sort by match score
            recommendations.sort(key=lambda x: x["match_score"], reverse=True)

            return jsonify(recommendations), 200

        return jsonify([]), 200

    except Exception as e:
        error_info = handle_exception(*sys.exc_info())
        return jsonify({"error": error_info}), 500


@app.route("/api/applicants/find-similar", methods=["POST"])
@cross_origin()
def find_similar():
    """
    Find similar applicants to a given profile.

    Input JSON Format:
    {
        "profile": {
            "name": "Test User",
            "age": 25,
            "experience": [
                {
                    "company": "Tech Corp",
                    "position": "Software Engineer",
                    "startYear": "2020-01-01",
                    "endYear": "2022-12-31"
                }
            ],
            "weekly_hours": 40,
            "languages": [
                {"language": "English", "proficiency": "Native"},
                {"language": "Spanish", "proficiency": "Intermediate"}
            ],
            "skills": [
                {"skillName": "Python", "proficiency": "Advanced"},
                {"skillName": "Java", "proficiency": "Intermediate"}
            ],
            "tags": ["IT", "Marketing"],
            "seeking": "jobs",
            "education": {
                "institutionName": "MIT",
                "institutionType": "college",
                "major": "Computer Science",
                "cgpa": "3.8"
            }
        }
    }

    Returns:
    [
        {
            "id": "507f1f77bcf86cd799439011",
            "name": "Similar Candidate",
            "similarity_score": 88.5,
            "match_factors": {
                "skills_match": 3,
                "language_match": 2,
                "tags_match": 2,
                "education_level_match": true,
                "experience_gap": 1
            },
            "profile": {
                "age": 26,
                "experience": 3,
                "education": {
                    "institutionName": "Stanford",
                    "institutionType": "college",
                    "major": "Computer Science",
                    "cgpa": "3.9"
                },
                "skills": [...],
                "languages": [...],
                "tags": [...],
                "seeking": "jobs"
            }
        },
        // ... more similar profiles
    ]

    Error Response:
    {
        "error": {
            "type": "ValueError",
            "message": "Invalid profile format",
            "traceback": "..."
        }
    }
    """
    try:
        data = request.get_json()
        target_profile = data.get("profile", {})

        # Validate input
        if not target_profile:
            return jsonify({"error": "Profile is required"}), 400

        # Build query to find potential matches
        base_query = {
            "isResumeParsed": True  # Only consider profiles that have been parsed
        }

        # Add filters based on target profile
        if target_profile.get("seeking"):
            base_query["profile.seeking"] = target_profile["seeking"]

        if target_profile.get("education", {}).get("institutionType"):
            base_query["profile.education.institutionType"] = target_profile[
                "education"
            ]["institutionType"]

        # Get potential matches
        potential_matches = list(applicants_collection.find(base_query))

        if not potential_matches:
            return jsonify([]), 200

        # Calculate similarity scores
        similarity_results = []
        for candidate in potential_matches:
            # Skip if comparing with self
            if str(candidate.get("_id")) == str(target_profile.get("_id")):
                continue

            # Original similarity calculation
            similarity_score = calculate_detailed_similarity(target_profile, candidate)

            # Apply communication adjustment to the similarity score
            adjusted_score = adjust_similarity_with_communication(
                similarity_score["total_score"], candidate
            )

            # Use the adjusted score
            if adjusted_score > 50:  # Minimum similarity threshold
                similarity_results.append(
                    {
                        "id": str(candidate["_id"]),
                        "name": candidate.get("name", ""),
                        "similarity_score": adjusted_score,  # Use adjusted score as the main score
                        "original_similarity_score": similarity_score[
                            "total_score"
                        ],  # Include original for reference
                        "communication_factor": {
                            "score": get_communication_score(candidate),
                            "impact": (
                                "adjusted"
                                if get_communication_score(candidate) is not None
                                else "none"
                            ),
                        },
                        "match_factors": {
                            "skills_match": similarity_score["skills_match"],
                            "education_match": similarity_score["education_match"],
                            "experience_match": similarity_score["experience_match"],
                            "language_match": similarity_score["language_match"],
                            "tag_match": similarity_score["tag_match"],
                        },
                        "profile": {
                            "age": candidate["profile"].get("age"),
                            "education": candidate["profile"].get("education"),
                            "experience": candidate["profile"].get("experience"),
                            "skills": candidate["profile"].get("skills"),
                            "languages": candidate["profile"].get("languages"),
                            "tags": candidate["profile"].get("tags"),
                            "seeking": candidate["profile"].get("seeking"),
                        },
                    }
                )

        # Sort by similarity score
        similarity_results.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Return top 10 matches
        return jsonify(similarity_results[:10]), 200

    except Exception as e:
        error_info = handle_exception(*sys.exc_info())
        return jsonify({"error": error_info}), 500


def calculate_detailed_similarity(target_profile, candidate_profile):
    """Calculate detailed similarity scores between two profiles"""
    scores = {
        "skills_match": calculate_skills_similarity(
            target_profile.get("skills", []),
            candidate_profile["profile"].get("skills", []),
        ),
        "education_match": calculate_education_similarity(
            target_profile.get("education", {}),
            candidate_profile["profile"].get("education", {}),
        ),
        "experience_match": calculate_experience_similarity(
            target_profile.get("experience", []),
            candidate_profile["profile"].get("experience", []),
        ),
        "language_match": calculate_language_similarity(
            target_profile.get("languages", []),
            candidate_profile["profile"].get("languages", []),
        ),
        "tag_match": calculate_tag_similarity(
            target_profile.get("tags", []), candidate_profile["profile"].get("tags", [])
        ),
        "age_match": calculate_age_similarity(
            target_profile.get("age"), candidate_profile["profile"].get("age")
        ),
        "personality_match": calculate_personality_similarity(
            target_profile.get("personalityBlueprint", []),
            candidate_profile["profile"].get("personalityBlueprint", []),
        ),
    }

    # Define weights with matching keys
    weights = {
        "skills_match": 0.25,
        "education_match": 0.20,
        "experience_match": 0.15,
        "language_match": 0.10,
        "tag_match": 0.10,
        "age_match": 0.10,
        "personality_match": 0.10,
    }

    # Calculate total score
    total_score = sum(scores[key] * weights[key] for key in scores.keys()) * 100

    return {
        "total_score": round(total_score, 2),
        **scores,  # Include all individual scores in the result
    }


def calculate_personality_similarity(profile1, profile2):
    """Calculate similarity between personality profiles based on OCEAN trait scores"""
    # Check if both profiles have traitScores
    trait_scores1 = profile1.get("traitScores", None)
    trait_scores2 = profile2.get("traitScores", None)

    if not trait_scores1 or not trait_scores2:
        # Fall back to old method if traitScores not available
        if profile1.get("personalityBlueprint") and profile2.get(
            "personalityBlueprint"
        ):
            # Legacy personality blueprint comparison logic
            responses1 = {
                q["questionId"]: q["selectedOption"]
                for q in profile1.get("personalityBlueprint", [])
            }
            responses2 = {
                q["questionId"]: q["selectedOption"]
                for q in profile2.get("personalityBlueprint", [])
            }

            total_questions = len(set(responses1.keys()) & set(responses2.keys()))
            if total_questions == 0:
                return 0.0

            similarity_sum = 0
            for qid in responses1:
                if qid in responses2:
                    diff = abs(responses1[qid] - responses2[qid])
                    similarity = 1 - (diff / 4)
                    similarity_sum += similarity

            return similarity_sum / total_questions
        return 0.0

    # New OCEAN-based calculation
    traits = ["o", "c", "e", "a", "n"]
    total_diff = 0

    for trait in traits:
        # Get trait scores, defaulting to 5 (middle value) if missing
        score1 = trait_scores1.get(trait, 5)
        score2 = trait_scores2.get(trait, 5)

        # Calculate absolute difference for this trait
        diff = abs(score1 - score2)
        total_diff += diff

    # Calculate average difference across all 5 traits
    avg_diff = total_diff / 5

    # Convert to similarity score (10 - avg_diff) / 10 * 100%
    similarity = (10 - avg_diff) / 10

    # Ensure the score is between 0 and 1
    return max(0, min(1, similarity))


def calculate_skills_similarity(skills1, skills2):
    """Calculate similarity between skill sets."""
    if not skills1 or not skills2:
        return 0.0

    skills1_names = set(s["skillName"].lower() for s in skills1)
    skills2_names = set(s["skillName"].lower() for s in skills2)

    intersection = len(skills1_names.intersection(skills2_names))
    union = len(skills1_names.union(skills2_names))

    return intersection / union if union > 0 else 0.0


def calculate_education_similarity(edu1, edu2):
    """Calculate similarity between education backgrounds."""
    if not edu1 or not edu2:
        return 0.0

    score = 0.0
    total_factors = 4

    # Same institution type
    if edu1.get("institutionType") == edu2.get("institutionType"):
        score += 1.0

    # Similar major
    if edu1.get("major", "").lower() == edu2.get("major", "").lower():
        score += 1.0

    # Close CGPA
    try:
        cgpa1 = float(edu1.get("cgpa", 0))
        cgpa2 = float(edu2.get("cgpa", 0))
        if abs(cgpa1 - cgpa2) <= 0.5:
            score += 1.0
    except (ValueError, TypeError):
        total_factors -= 1

    # Close graduation years
    try:
        year1 = int(edu1.get("endYear", 0))
        year2 = int(edu2.get("endYear", 0))
        if abs(year1 - year2) <= 2:
            score += 1.0
    except (ValueError, TypeError):
        total_factors -= 1

    return score / total_factors if total_factors > 0 else 0.0


def calculate_experience_similarity(exp1, exp2):
    """Calculate similarity between experience sets."""
    if not exp1 or not exp2:
        return 0.0

    # Compare experience duration and roles
    total_similarity = 0
    comparisons = 0

    for e1 in exp1:
        for e2 in exp2:
            role_similarity = 0.0
            if e1.get("position", "").lower() == e2.get("position", "").lower():
                role_similarity = 1.0
            elif any(
                word in e1.get("position", "").lower()
                for word in e2.get("position", "").lower().split()
            ):
                role_similarity = 0.5

            total_similarity += role_similarity
            comparisons += 1

    return total_similarity / comparisons if comparisons > 0 else 0.0


def calculate_language_similarity(langs1, langs2):
    """Calculate similarity between language sets."""
    if not langs1 or not langs2:
        return 0.0

    langs1_set = set(l["language"].lower() for l in langs1)
    langs2_set = set(l["language"].lower() for l in langs2)

    intersection = len(langs1_set.intersection(langs2_set))
    union = len(langs1_set.union(langs2_set))

    return intersection / union if union > 0 else 0.0


def calculate_tag_similarity(tags1, tags2):
    """Calculate similarity between tag sets."""
    if not tags1 or not tags2:
        return 0.0

    tags1_set = set(t.lower() for t in tags1)
    tags2_set = set(t.lower() for t in tags2)

    intersection = len(tags1_set.intersection(tags2_set))
    union = len(tags1_set.union(tags2_set))

    return intersection / union if union > 0 else 0.0


def calculate_age_similarity(age1, age2):
    """Calculate similarity between ages."""
    if not age1 or not age2:
        return 0.0

    try:
        age_diff = abs(int(age1) - int(age2))
        if age_diff <= 2:
            return 1.0
        elif age_diff <= 5:
            return 0.5
        else:
            return 0.0
    except (ValueError, TypeError):
        return 0.0


@app.route("/api/applicants/find", methods=["GET", "POST"])
@cross_origin()
def get_applicant():
    """
    Get detailed information about a specific applicant.

    Input JSON Format:
    {
        "id": "507f1f77bcf86cd799439011"
    }

    Returns:
    {
        "_id": "507f1f77bcf86cd799439011",
        "name": "John Doe",
        "username": "johndoe",
        "email": "john.doe@example.com",
        "type": "fresher",
        "profile": {
            "phoneNumber": "+1234567890",
            "age": 25,
            "education": {
                "institutionName": "MIT",
                "institutionType": "college",
                "major": "Computer Science",
                "cgpa": "3.8",
                "startYear": 2019,
                "endYear": 2023,
                "year": 4
            },
            "experience": [...],
            "projects": [...],
            "skills": [...],
            "languages": [...],
            "tags": [...],
            "seeking": "jobs"
        },
        "createdAt": "2023-01-01T00:00:00Z",
        "updatedAt": "2023-01-01T00:00:00Z"
    }

    Error Response:
    {
        "error": {
            "type": "ValueError",
            "message": "Invalid applicant ID",
            "traceback": "..."
        }
    }
    """
    try:
        data = request.get_json()
        applicant_id = data.get("id", {})
        applicant = applicants_collection.find_one({"_id": ObjectId(applicant_id)})

        if not applicant:
            return jsonify({"error": "Applicant not found"}), 404

        # Remove sensitive information
        applicant.pop("password", None)
        applicant.pop("otp", None)
        applicant.pop("otpExpiry", None)

        # Convert ObjectId to string
        applicant["_id"] = str(applicant["_id"])

        return jsonify(applicant), 200

    except Exception as e:
        error_info = handle_exception(*sys.exc_info())
        return jsonify({"error": error_info}), 500


DEEPSEEK_API_KEY = "sk-d36413ecc68f4350995b8531ec8ceb4e"
# if not DEEPSEEK_API_KEY:
#     raise ValueError("DEEPSEEK_API_KEY environment variable not set")

try:
    deepseek_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",  # Updated base URL
    )
except Exception as e:
    print(f"Error initializing DeepSeek client: {e}")
    raise


def _call_deepseek_api(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Make API call to Deepseek with retry logic"""
    max_retries = 3
    retry_delay = 1
    response = None
    last_error = None

    deepseek_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",  # Updated base URL
    )

    for attempt in range(max_retries):
        try:
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=0.3,
            )
            return {
                "choices": [
                    {"message": {"content": response.choices[0].message.content}}
                ]
            }
        except Exception as e:
            last_error = str(e)
            if attempt == max_retries:
                raise Exception(
                    f"Failed to get response from DeepSeek API after {max_retries} retries. Last error: {last_error}"
                )
            time.sleep(retry_delay)
        finally:
            if response is None and attempt == max_retries:
                raise Exception(
                    f"Maximum retries reached. Failed to get valid response from DeepSeek API. Last error: {last_error}"
                )


def _clean_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and parse the API response"""
    try:
        content = response["choices"][0]["message"]["content"]

        # Remove markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        # Clean the content
        content = (
            content.strip()  # Remove leading/trailing whitespace
            .replace("\n", "")  # Remove newlines
            .replace("\r", "")  # Remove carriage returns
            .replace("\t", "")  # Remove tabs
        )

        # Remove multiple spaces
        while "  " in content:
            content = content.replace("  ", " ")

        # Remove any potential markdown or text formatting
        if content.startswith("`") and content.endswith("`"):
            content = content[1:-1]

        # Attempt to parse the cleaned JSON
        return {
            "parsed_data": json.loads(content),
            "raw_content": content,  # Store the cleaned content
        }
    except json.JSONDecodeError as e:
        return {
            "error": f"JSON parsing error: {str(e)}",
            "raw_content": content,
            "original_content": response["choices"][0]["message"]["content"],
            "parsing_failed": True,
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "raw_response": response,
            "parsing_failed": True,
        }


@app.route("/api/candidates/construct", methods=["POST"])
@cross_origin()
def construct_candidate():
    """
    Construct an ideal candidate profile based on a job description using DeepSeek API.
    """
    try:
        # Verify API key is set
        if not DEEPSEEK_API_KEY:
            return (
                jsonify({"error": "DeepSeek API key not configured", "success": False}),
                500,
            )

        job_data = request.get_json()

        # Construct the system prompt
        system_prompt = """You are an expert HR professional specializing in creating ideal candidate profiles based on job descriptions.
        Your task is to analyze the job requirements and create a detailed candidate profile that would be perfect for the position.

        The output should be a JSON object with this exact structure:
        {
            "name": "Ideal Candidate",
            "profile": {
                "age": number (between 20-45),
                "education": {
                    "institutionName": string,
                    "institutionType": "college",
                    "major": string,
                    "cgpa": string (format: "3.8"),
                    "year": number (1-4)
                },
                "experience": [
                    {
                        "company": string,
                        "position": string,
                        "startYear": string (YYYY format),
                        "endYear": string (YYYY format),
                        "description": string
                    }
                ],
                "skills": [
                    {
                        "skillName": string,
                        "proficiency": string (one of: "Beginner", "Intermediate", "Advanced")
                    }
                ],
                "languages": [
                    {
                        "language": string,
                        "proficiency": string (one of: "Beginner", "Intermediate", "Advanced", "Native")
                    }
                ],
                "tags": array of strings (from: ["IT", "Marketing", "Finance and Accounting", "Sales", "Human Resources", "Legal", "Retail", "Customer Service"]),
                "seeking": string (one of: "internship", "jobs"),
                "traitScores": {
                    "o": number (between 1-10, representing Openness),
                    "c": number (between 1-10, representing Conscientiousness),
                    "e": number (between 1-10, representing Extraversion),
                    "a": number (between 1-10, representing Agreeableness),
                    "n": number (between 1-10, representing Neuroticism)
                },
                "personalityBlueprint": [
                    {
                        "questionId": "67540aa95bada2c0c1b5d518",
                        "selectedOption": number (1-5),
                        "trait": "Openness/Intuition"
                    },
                    {
                        "questionId": "67540aa95bada2c0c1b5d519",
                        "selectedOption": number (1-5),
                        "trait": "Conscientiousness"
                    },
                    {
                        "questionId": "67540aa95bada2c0c1b5d51a",
                        "selectedOption": number (1-5),
                        "trait": "Extraversion"
                    },
                    {
                        "questionId": "67540aa95bada2c0c1b5d51b",
                        "selectedOption": number (1-5),
                        "trait": "Agreeableness"
                    },
                    {
                        "questionId": "67540aa95bada2c0c1b5d51c",
                        "selectedOption": number (1-5),
                        "trait": "Emotional Stability"
                    },
                    {
                        "questionId": "67540aa95bada2c0c1b5d51d",
                        "selectedOption": number (1-5),
                        "trait": "Decision Making"
                    },
                    {
                        "questionId": "67540aa95bada2c0c1b5d51e",
                        "selectedOption": number (1-5),
                        "trait": "Workplace Behavior"
                    },
                    {
                        "questionId": "67540aa95bada2c0c1b5d51f",
                        "selectedOption": number (1-5),
                        "trait": "Risk Taking"
                    },
                    {
                        "questionId": "67540aa95bada2c0c1b5d520",
                        "selectedOption": number (1-5),
                        "trait": "Flexibility"
                    },
                    {
                        "questionId": "67540aa95bada2c0c1b5d521",
                        "selectedOption": number (1-5),
                        "trait": "Social Sensitivity"
                    }
                ]
            }
        }

        Personality Trait Scoring Guide:
        1. Openness/Intuition: 1 = Very Traditional -> 5 = Highly Innovative
        2. Conscientiousness: 1 = Flexible/Spontaneous -> 5 = Highly Organized
        3. Extraversion: 1 = Highly Introverted -> 5 = Highly Extraverted
        4. Agreeableness: 1 = Task-Focused -> 5 = People-Focused
        5. Emotional Stability: 1 = Reactive -> 5 = Resilient
        6. Decision Making: 1 = Purely Logical -> 5 = Highly Empathetic
        7. Workplace Behavior: 1 = Team Member -> 5 = Natural Leader
        8. Risk Taking: 1 = Very Cautious -> 5 = Risk-Embracing
        9. Flexibility: 1 = Structured -> 5 = Adaptable
        10. Social Sensitivity: 1 = Direct/Task-Oriented -> 5 = Diplomatic/People-Oriented

        OCEAN Trait Scoring Guidelines for traitScores (1-10 scale):
        - O (Openness): Higher scores indicate creativity, curiosity, and openness to new experiences
        - C (Conscientiousness): Higher scores indicate organization, dependability, and self-discipline
        - E (Extraversion): Higher scores indicate sociability, assertiveness, and energy
        - A (Agreeableness): Higher scores indicate cooperation, compassion, and consideration
        - N (Neuroticism): Higher scores indicate anxiety, emotional instability, and negative emotions

        For the job description provided:
        1. Select appropriate OCEAN trait scores (1-10) that would make the candidate successful in this role
        2. Ensure consistency between the traitScores and personalityBlueprint selections
        3. Adapt the scores to match job requirements (e.g., higher Conscientiousness for detail-oriented roles)

        Create a profile that:
        1. Matches the job requirements exactly
        2. Has realistic education and experience requirements
        3. Includes appropriate skills and proficiency levels
        4. Specifies required languages with proficiency levels
        5. Selects appropriate age range for the role
        6. Chooses personality traits that would excel in this role
        7. Uses only the specified tags
        8. Maintains internal consistency across all fields

        Return only the JSON object, with no additional text or formatting."""

        # Construct the user prompt
        user_prompt = f"""Create an ideal candidate profile for the following job:

        Position: {job_data.get('jobPosition', '')}
        Description: {job_data.get('description', '')}
        Experience Required: {job_data.get('experienceRequired', '')}
        Skills Required: {', '.join(job_data.get('skillsRequired', []))}
        Qualifications: {', '.join(job_data.get('qualifications', []))}
        Job Requirements: {', '.join(job_data.get('jobRequirements', []))}
        English Requirements: {job_data.get('englishRequirements', '')}
        Employment Type: {job_data.get('employmentType', '')}
        Location: {', '.join(job_data.get('location', []))}"""

        try:
            # Call DeepSeek API with retry logic
            api_response = _call_deepseek_api(system_prompt, user_prompt)
        except Exception as e:
            return (
                jsonify(
                    {"error": f"DeepSeek API call failed: {str(e)}", "success": False}
                ),
                500,
            )

        # Clean and parse the response
        cleaned_response = _clean_response(api_response)

        # Check if parsing failed
        if cleaned_response.get("parsing_failed"):
            return (
                jsonify(
                    {
                        "error": cleaned_response.get(
                            "error", "Failed to parse response"
                        ),
                        "success": False,
                        "raw_response": cleaned_response.get("raw_content"),
                        "api_response": api_response,
                    }
                ),
                500,
            )

        # Return the cleaned and parsed response
        return (
            jsonify(
                {
                    "success": True,
                    "data": cleaned_response["parsed_data"],
                    "raw_response": cleaned_response.get("raw_content"),
                }
            ),
            200,
        )

    except Exception as e:
        error_info = handle_exception(*sys.exc_info())
        return jsonify({"error": error_info, "success": False}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=app.config["DEBUG"]
    )
