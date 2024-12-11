from numpy import int8
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import logging
import traceback
import sys
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client["your_database_name"]
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
    """Convert MongoDB document to feature vector"""
    features = {
        "uid": str(applicant["_id"]),
        "name": applicant["name"],
        "age": applicant["profile"].get("age", 0),
        "experience": len(applicant["profile"].get("experience", [])),
        "weekly_hours": 40,  # Default
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

    # Add job type features
    features["seeking"] = applicant["profile"].get("seeking", "jobs")

    return features


def calculate_similarity(applicant1, applicant2):
    """Calculate similarity between two applicants"""
    features = ["age", "experience"] + LANGUAGES + TAGS

    vector1 = [applicant1[f] for f in features]
    vector2 = [applicant2[f] for f in features]

    similarity = cosine_similarity([vector1], [vector2])[0][0]
    return round(similarity * 100, 2)


def find_similar_applicants(target_applicant, k=5):
    """Find k most similar applicants"""
    all_applicants = list(applicants_collection.find({}))
    feature_vectors = [transform_mongodb_to_features(app) for app in all_applicants]

    # Filter by same job seeking type
    filtered_applicants = [
        app for app in feature_vectors if app["seeking"] == target_applicant["seeking"]
    ]

    similarities = [
        (app, calculate_similarity(target_applicant, app))
        for app in filtered_applicants
    ]

    return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        # Test MongoDB connection
        applicants_collection.find_one()
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/applicants/compare", methods=["POST"])
@cross_origin()
def compare_applicants():
    """Compare two applicants"""
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

        # Calculate differences
        differences = []

        # Experience difference
        exp_diff = len(applicant1["profile"].get("experience", [])) - len(
            applicant2["profile"].get("experience", [])
        )
        if exp_diff != 0:
            differences.append(
                {
                    "factor": "experience",
                    "explanation": f"{applicant1['name']} has {abs(exp_diff)} "
                    f"{'more' if exp_diff > 0 else 'less'} years of experience",
                }
            )

        # Language differences
        langs1 = set(
            l["language"].lower() for l in applicant1["profile"].get("languages", [])
        )
        langs2 = set(
            l["language"].lower() for l in applicant2["profile"].get("languages", [])
        )

        if langs1 != langs2:
            unique1 = langs1 - langs2
            unique2 = langs2 - langs1
            if unique1:
                differences.append(
                    {
                        "factor": "languages",
                        "explanation": f"{applicant1['name']} additionally knows: {', '.join(unique1)}",
                    }
                )
            if unique2:
                differences.append(
                    {
                        "factor": "languages",
                        "explanation": f"{applicant2['name']} additionally knows: {', '.join(unique2)}",
                    }
                )

        response = {"similarity_score": similarity_score, "differences": differences}

        return jsonify(response), 200

    except Exception as e:
        error_info = handle_exception(*sys.exc_info())
        return jsonify({"error": error_info}), 500


@app.route("/api/applicants/recommend", methods=["POST"])
@cross_origin()
def recommend_applicants():
    """Get applicant recommendations based on requirements"""
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
            query["profile.experience"] = {
                "$size": {"$gte": requirements["min_experience"]}
            }

        # Skills requirement
        if "required_skills" in requirements:
            query["profile.skills.skillName"] = {"$in": requirements["required_skills"]}

        # Tags requirement
        if "tags" in requirements:
            query["profile.tags"] = {"$in": requirements["tags"]}

        # Get matching applicants
        matching_applicants = list(applicants_collection.find(query))

        # Transform to feature vectors for similarity calculation
        if matching_applicants:
            feature_vectors = [
                transform_mongodb_to_features(app) for app in matching_applicants
            ]

            # Calculate similarities between all pairs
            recommendations = []
            for i, app in enumerate(matching_applicants):
                scores = []
                for j, other_app in enumerate(matching_applicants):
                    if i != j:
                        similarity = calculate_similarity(
                            feature_vectors[i], feature_vectors[j]
                        )
                        scores.append(similarity)

                avg_similarity = sum(scores) / len(scores) if scores else 0

                recommendations.append(
                    {
                        "id": str(app["_id"]),
                        "name": app["name"],
                        "match_score": round(avg_similarity, 2),
                        "profile": {
                            "age": app["profile"].get("age"),
                            "experience": len(app["profile"].get("experience", [])),
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
    """Find similar applicants to a given profile"""
    try:
        data = request.get_json()
        profile = data.get("profile", {})

        # Convert incoming profile to feature format
        target_features = {
            "uid": "temp",
            "name": profile.get("name", "Temporary"),
            "age": profile.get("age", 0),
            "experience": len(profile.get("experience", [])),
            "weekly_hours": profile.get("weekly_hours", 40),
        }

        # Add language features
        for lang in LANGUAGES:
            target_features[lang] = int(
                lang.lower()
                in [l["language"].lower() for l in profile.get("languages", [])]
            )

        # Add tags features
        for tag in TAGS:
            target_features[tag] = int(tag in profile.get("tags", []))

        target_features["seeking"] = profile.get("seeking", "jobs")

        # Find similar applicants
        similar_applicants = find_similar_applicants(target_features)

        # Format response
        results = []
        for app, score in similar_applicants:
            applicant = applicants_collection.find_one({"_id": ObjectId(app["uid"])})
            if applicant:
                results.append(
                    {
                        "id": str(applicant["_id"]),
                        "name": applicant["name"],
                        "similarity_score": score,
                        "profile": {
                            "age": applicant["profile"].get("age"),
                            "experience": len(
                                applicant["profile"].get("experience", [])
                            ),
                            "languages": [
                                l["language"]
                                for l in applicant["profile"].get("languages", [])
                            ],
                            "skills": [
                                s["skillName"]
                                for s in applicant["profile"].get("skills", [])
                            ],
                            "tags": applicant["profile"].get("tags", []),
                        },
                    }
                )

        return jsonify(results), 200

    except Exception as e:
        error_info = handle_exception(*sys.exc_info())
        return jsonify({"error": error_info}), 500


@app.route("/api/applicants/<applicant_id>", methods=["GET"])
@cross_origin()
def get_applicant(applicant_id):
    """Get detailed information about a specific applicant"""
    try:
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


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
