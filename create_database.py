from pymongo import MongoClient
from datetime import datetime, timedelta
import bcrypt
from bson import ObjectId

from pymongo.server_api import ServerApi

# Connect to MongoDB
uri = "mongodb+srv://smitshahcloudboost:1234@cluster0.45ng0.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi("1"))
db = client["internlink_deployment_dummy"]
applicants_collection = db["applicants"]

# Clear existing data
applicants_collection.delete_many({})

# Sample data
sample_applicants = [
    {
        "_id": ObjectId(),
        "name": "John Doe",
        "username": "johndoe",
        "email": "john.doe@example.com",
        "authMethods": [{"type": "email", "verified": True}],
        "gender": "male",
        "password": bcrypt.hashpw("password123".encode("utf-8"), bcrypt.gensalt()),
        "type": "fresher",
        "isResumeParsed": True,
        "profile": {
            "phoneNumber": "+1234567890",
            "age": 22,
            "education": {
                "institutionName": "MIT",
                "institutionType": "college",
                "major": "Computer Science",
                "cgpa": "3.8",
                "startYear": 2019,
                "endYear": 2023,
                "year": 4,
            },
            "experience": [
                {
                    "company": "Tech Corp",
                    "position": "Software Engineer Intern",
                    "startYear": datetime(2022, 6, 1),
                    "endYear": datetime(2022, 12, 31),
                }
            ],
            "projects": [
                {
                    "projectName": "AI Chatbot",
                    "description": "Built an AI-powered chatbot using Python and TensorFlow",
                    "status": "completed",
                    "demoLink": "https://github.com/johndoe/chatbot",
                    "startYear": datetime(2022, 1, 1),
                    "endYear": datetime(2022, 5, 1),
                }
            ],
            "skills": [
                {"skillName": "Python", "proficiency": "Advanced"},
                {"skillName": "Machine Learning", "proficiency": "Intermediate"},
            ],
            "languages": [
                {"language": "English", "proficiency": "Native"},
                {"language": "Hindi", "proficiency": "Intermediate"},
            ],
            "tags": ["IT", "Marketing"],
            "seeking": "internship",
        },
        "createdAt": datetime.now(),
        "updatedAt": datetime.now(),
    },
    {
        "_id": ObjectId(),
        "name": "Jane Smith",
        "username": "janesmith",
        "email": "jane.smith@example.com",
        "googleId": "google123",
        "authMethods": [{"type": "google", "verified": True}],
        "gender": "female",
        "type": "college Student",
        "isResumeParsed": True,
        "profile": {
            "phoneNumber": "+9876543210",
            "age": 25,
            "education": {
                "institutionName": "Stanford University",
                "institutionType": "college",
                "major": "Data Science",
                "cgpa": "3.9",
                "startYear": 2018,
                "endYear": 2022,
                "year": 4,
            },
            "experience": [
                {
                    "company": "Data Analytics Co",
                    "position": "Data Scientist",
                    "startYear": datetime(2022, 1, 1),
                    "endYear": datetime.now(),
                }
            ],
            "skills": [
                {"skillName": "Python", "proficiency": "Expert"},
                {"skillName": "Data Analysis", "proficiency": "Advanced"},
            ],
            "languages": [
                {"language": "English", "proficiency": "Native"},
                {"language": "Spanish", "proficiency": "Advanced"},
            ],
            "tags": ["IT", "Finance and Accounting"],
            "seeking": "jobs",
        },
        "createdAt": datetime.now(),
        "updatedAt": datetime.now(),
    },
    {
        "_id": ObjectId(),
        "name": "Raj Patel",
        "username": "rajpatel",
        "email": "raj.patel@example.com",
        "linkedInId": "linkedin123",
        "authMethods": [{"type": "linkedin", "verified": True}],
        "gender": "male",
        "type": "fresher",
        "isResumeParsed": True,
        "profile": {
            "phoneNumber": "+91987654321",
            "age": 23,
            "education": {
                "institutionName": "IIT Bombay",
                "institutionType": "college",
                "major": "Electronics",
                "cgpa": "8.5",
                "startYear": 2019,
                "endYear": 2023,
                "year": 4,
            },
            "skills": [
                {"skillName": "Circuit Design", "proficiency": "Advanced"},
                {"skillName": "VLSI", "proficiency": "Intermediate"},
            ],
            "languages": [
                {"language": "English", "proficiency": "Advanced"},
                {"language": "Hindi", "proficiency": "Native"},
                {"language": "Gujarati", "proficiency": "Native"},
            ],
            "tags": ["IT", "Sales"],
            "seeking": "jobs",
        },
        "createdAt": datetime.now(),
        "updatedAt": datetime.now(),
    },
    {
        "_id": ObjectId(),
        "name": "Maria Garcia",
        "username": "mariagarcia",
        "email": "maria.garcia@example.com",
        "authMethods": [{"type": "email", "verified": True}],
        "gender": "female",
        "password": bcrypt.hashpw("password456".encode("utf-8"), bcrypt.gensalt()),
        "type": "college Student",
        "isResumeParsed": True,
        "profile": {
            "phoneNumber": "+34612345678",
            "age": 24,
            "education": {
                "institutionName": "Universidad de Barcelona",
                "institutionType": "college",
                "major": "Business Administration",
                "cgpa": "8.2",
                "startYear": 2019,
                "endYear": 2023,
                "year": 4,
            },
            "skills": [
                {"skillName": "Marketing Strategy", "proficiency": "Advanced"},
                {"skillName": "Social Media Marketing", "proficiency": "Expert"},
            ],
            "languages": [
                {"language": "English", "proficiency": "Advanced"},
                {"language": "Spanish", "proficiency": "Native"},
            ],
            "tags": ["Marketing", "Sales"],
            "seeking": "internship",
        },
        "createdAt": datetime.now(),
        "updatedAt": datetime.now(),
    },
    {
        "_id": ObjectId(),
        "name": "Alex Chen",
        "username": "alexchen",
        "email": "alex.chen@example.com",
        "googleId": "google456",
        "authMethods": [{"type": "google", "verified": True}],
        "gender": "male",
        "type": "fresher",
        "isResumeParsed": True,
        "profile": {
            "phoneNumber": "+6591234567",
            "age": 26,
            "education": {
                "institutionName": "National University of Singapore",
                "institutionType": "college",
                "major": "Finance",
                "cgpa": "3.7",
                "startYear": 2018,
                "endYear": 2022,
                "year": 4,
            },
            "experience": [
                {
                    "company": "Investment Bank Corp",
                    "position": "Financial Analyst",
                    "startYear": datetime(2022, 6, 1),
                    "endYear": datetime.now(),
                }
            ],
            "skills": [
                {"skillName": "Financial Modeling", "proficiency": "Expert"},
                {"skillName": "Data Analysis", "proficiency": "Advanced"},
            ],
            "languages": [
                {"language": "English", "proficiency": "Native"},
                {"language": "Mandarin", "proficiency": "Native"},
            ],
            "tags": ["Finance and Accounting", "Sales"],
            "seeking": "jobs",
        },
        "createdAt": datetime.now(),
        "updatedAt": datetime.now(),
    },
]

# Insert sample data
try:
    result = applicants_collection.insert_many(sample_applicants)
    print(f"Successfully inserted {len(result.inserted_ids)} candidates")
    print("\nInserted IDs:")
    for id in result.inserted_ids:
        print(id)
except Exception as e:
    print(f"An error occurred: {e}")

# Verify the insertion
count = applicants_collection.count_documents({})
print(f"\nTotal documents in collection: {count}")

# Print a sample document (first one)
print("\nSample document:")
print(applicants_collection.find_one({"name": "John Doe"}))
