const mongoose = require("mongoose");

const JobSchema = new mongoose.Schema({
  jobPosition: {
    type: String,
    required: [true, "Job position is required"],
    trim: true,
  },
  description: {
    type: String,
    required: [true, "Job description is required"],
  },
  employer: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User", // Reference to the User schema for employer details
    required: [true, "Employer is required"],
  },
  companyName: {
    type: String,
  },
  location: [
    {
      type: String,
      required: [true, "Job location is required"],
    },
  ],
  // employmentType: {
  //     type: String,
  //     enum: ['full-time', 'part-time', 'contract', 'internship', 'temporary', 'freelance'],
  //     required: [true, 'Employment type is required'],
  // },
  employmentType: {
    type: String,
    // enum: [
    //   "Full-Time",
    //   "Full-time",
    //   "Part-Time",
    //   "Part-time",
    //   "Contract",
    //   "Internship",
    //   "Temporary",
    //   "Freelance",
    //   "full-time", // Old format
    //   "part-time", // Old format
    //   "contract",  // Old format
    //   "internship", // Old format
    // ],
  },
  
  experienceRequired: {
    type: String,
    // enum: [
    //   "0-1 years",
    //   "1-2 years",
    //   "2-4 years",
    //   "4-6 years",
    //   "6+ years",
    //   "Other",
    // ],
    default: "Other", // Allow invalid values to be stored as "Other"
  },
  personalityType : {type: String},
  salary: {
    min: {
      type: Number,
      required: false, // Optional, as some jobs don't disclose salary
    },
    max: {
      type: Number,
      required: false,
    },
    currency: {
      type: String,
      default: "USD", // Default currency is USD
    },
  },
  skillsRequired: [
    {
      type: String,
    },
  ],
  qualifications: {
    type: mongoose.Schema.Types.Mixed, // Accept either format
    validate: {
      validator: function (v) {
        return (
          typeof v === "string" || // Old format
          (Array.isArray(v) && v.every((q) => typeof q === "string")) // New format
        );
      },
      message: "Qualifications must be a string or an array of strings.",
    },
  },
  
  jobRequirements: [{ type: String }],
  englishRequirements: {
    type: String,
    // enum: ["basic", "good", "fluent"],
    default: "basic", // Default to "basic" if not provided
  },
  // experienceRequired: {
  //     type: String, // E.g., "2-4 years" or "Fresher"
  //     required: false,
  //     enum: ['fresher', 'experienced']
  // },
  benefits: [{ type: String }],
  applicants: [
    {
      applicantId: { type: mongoose.Schema.Types.ObjectId, ref: "Applicant" },
      appliedAt: { type: Date, default: Date.now },
      compatibilityScore : {type: Number, required: false},
      status: {
        type: String,
        // enum: ["in-progress", "selected", "rejected", "shortlisted"], // Include "shortlisted"
        default: "in-progress",
      },
      
      interview: {
        completed: { type: Boolean, default: false },
        technicalSkills: {
          type: String,
          // enum: ["excellent", "good", "lacks the necessary skills"],
          default: "good",
        },
        problemSolving: {
          type: String,
          // enum: ["excellent", "poor", "good"],
          default: "good",
        },
        adaptability: {
          type: String,
          // enum: ["adaptable", "somewhat rigid", "rigid"],
          default: "adaptable",
        },
      },
    },
  ],
  paymentType: [
    { type: String, 
      // enum: ["fixed", "fixed+incentive", "incentive"] 
    },
  ],
  isRemote: {
    type: Boolean,
    default: false,
  },
  isNightShift: {
    type: Boolean,
    default: false,
  },
  fees: {
    required: { type: Boolean, default: false },
    type: { type: String, 
      // enum: ["Joining", "Deposit"], 
      default: null },
    amount: { type: Number, default: 0 }, // Default value for missing amount
  },

  postedAt: {
    type: Date,
    default: Date.now,
  },
  expiresAt: {
    type: Date, // Expiry date for the job post
  },
  status: {
    type: String,
    // enum: ["Active", "Closed"],
    default: "Active",
  },
  tags: [{ type: String }],
  image: {
    type: String, // URL or path to the company's image/logo
    required: false,
  },
  imageArray: [
    {
      type: String, //URLs of image assets to display on the frontend
    },
  ],
  views: [
    {
      viewedAt: { type: Date, default: Date.now },
      viewerId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "Applicant",
        required: false,
      },
    },
  ],
  viewCount: { type: Number, default: 0 },
  communicationPreference: {
    type: String,
    // enum: ["applicant", "user"],
    default: "applicant", // Default to "applicant" for missing values
  },
  interviewType: {
    type: String,
    // enum: ["offline", "online"],
    default: "offline", // Default to "offline" for old documents
  },
  city: { type: String, default: "" },
  updatedBy: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
  updatedAt: { type: Date, default: Date.now },
},{timestamps: true});

module.exports = mongoose.model("Job", JobSchema);
