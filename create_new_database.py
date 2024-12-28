from pymongo import MongoClient
from datetime import datetime, timedelta
import bcrypt
from bson import ObjectId
from pymongo.server_api import ServerApi

# MongoDB connection
uri = "mongodb+srv://smitshahcloudboost:1234@cluster0.45ng0.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi("1"))
db = client["internlink_deployment_dummy"]  # Using your database name
applicants_collection = db["applicants_2"]


def generate_and_insert_candidates():
    # Clear existing data
    applicants_collection.delete_many({})

    # Sample data pools
    names = [
        "Alex Thompson",
        "Sarah Chen",
        "Michael O'Connor",
        "Priya Patel",
        "James Wilson",
        "Maria Garcia",
        "David Kim",
        "Emma Brown",
        "Hassan Ahmed",
        "Olivia Martinez",
    ]

    companies = [
        "Google",
        "Microsoft",
        "Amazon",
        "Apple",
        "Meta",
        "IBM",
        "Intel",
        "Adobe",
        "Salesforce",
        "Oracle",
    ]

    universities = [
        "Stanford University",
        "MIT",
        "Carnegie Mellon",
        "UC Berkeley",
        "Georgia Tech",
        "University of Michigan",
        "Caltech",
        "University of Illinois",
        "Cornell University",
        "Harvard University",
    ]

    sample_candidates = []

    for i in range(10):
        # Calculate dates
        created_date = datetime.now() - timedelta(days=i * 10)
        updated_date = created_date + timedelta(days=5)

        candidate = {
            "_id": ObjectId(),
            "name": names[i],
            "username": names[i].lower().replace(" ", ""),
            "email": f"{names[i].lower().replace(' ', '.')}@example.com",
            "gender": "Male" if i % 2 == 0 else "Female",
            "password": (
                bcrypt.hashpw(f"password{i}123".encode("utf-8"), bcrypt.gensalt())
                if i % 2 == 0
                else None
            ),
            "type": "fresher" if i % 2 == 0 else "college Student",
            "isResumeParsed": True,
            "createdAt": created_date,
            "updatedAt": updated_date,
            "swipes": 5,
            "lastSwipeDate": datetime.now() - timedelta(days=1),
            "profile": {
                "phoneNumber": f"+1{str(i)*8}",
                "age": 22 + i,
                "education": {
                    "institutionName": universities[i],
                    "institutionType": "college",
                    "major": (
                        "Computer Science" if i % 2 == 0 else "Business Administration"
                    ),
                    "cgpa": f"{3.5 + (i/10):.2f}",
                    "startYear": 2019 + i,
                    "endYear": 2023 + i,
                    "year": 4,
                },
                "experience": [
                    {
                        "company": companies[i],
                        "position": (
                            "Software Engineer" if i % 2 == 0 else "Product Manager"
                        ),
                        "startYear": datetime(2022 + i, 1, 1),
                        "endYear": datetime(2023 + i, 12, 31),
                        "description": f"Worked on various projects at {companies[i]}",
                    }
                ],
                "projects": [
                    {
                        "projectName": f"Project {j+1}",
                        "description": f"Innovative project at {universities[i]}",
                        "status": "completed",
                        "demoLink": f"https://github.com/{names[i].lower().replace(' ', '')}/project{j+1}",
                        "startYear": datetime(2022, 1, 1),
                        "endYear": datetime(2022, 6, 1),
                    }
                    for j in range(2)
                ],
                "skills": [
                    {"skillName": "Python", "proficiency": "Advanced"},
                    {"skillName": "Java", "proficiency": "Intermediate"},
                    {"skillName": "Machine Learning", "proficiency": "Beginner"},
                    {"skillName": "React", "proficiency": "Advanced"},
                ],
                "languages": [
                    {"language": "English", "proficiency": "Native"},
                    {"language": "Spanish", "proficiency": "Intermediate"},
                    {"language": "Mandarin", "proficiency": "Beginner"},
                ],
                "tags": (
                    ["IT", "Marketing"] if i % 2 == 0 else ["Human Resources", "Sales"]
                ),
                "seeking": "internship" if i % 2 == 0 else "jobs",
            },
            "authMethods": [
                {
                    "_id": ObjectId(),
                    "type": "email" if i % 2 == 0 else "google",
                    "verified": True,
                }
            ],
        }

        # Add optional fields based on auth method
        if i % 2 != 0:  # Add googleId for Google auth
            candidate["googleId"] = f"google{i}123456789"

        # Add personality blueprint
        candidate["profile"]["personalityBlueprint"] = [
            {
                "_id": ObjectId(),
                "questionId": f"67540aa95bada2c0c1b5d5{j}",
                "selectedOption": j % 4,
            }
            for j in range(10)
        ]

        # Add applied jobs
        candidate["appliedJobs"] = [
            {
                "_id": ObjectId(),
                "jobId": str(ObjectId()),
                "appliedAt": (created_date + timedelta(days=j)).isoformat(),
                "status": "in-progress",
            }
            for j in range(3)
        ]

        sample_candidates.append(candidate)

    try:
        # Insert the candidates
        result = applicants_collection.insert_many(sample_candidates)
        print(f"Successfully inserted {len(result.inserted_ids)} candidates")
        print("\nInserted IDs:")
        for id in result.inserted_ids:
            print(id)

        # Verify the insertion
        count = applicants_collection.count_documents({})
        print(f"\nTotal documents in collection: {count}")

        # Print a sample document
        print("\nSample document:")
        print(applicants_collection.find_one({"name": names[0]}))

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    generate_and_insert_candidates()
