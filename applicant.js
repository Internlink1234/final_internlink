const mongoose = require("mongoose");
const validator = require("validator");
const bcrypt = require("bcryptjs");

const ApplicantSchema = new mongoose.Schema({
  name: {
    type: String,
    trim: true,
    default: "",
  },
  username: {
    type: String,
    trim: true,
    default: "",
  },
  email: {
    type: String,
    // required: [true, "Email is required"],
    unique: true,
    lowercase: true,
    validate: {
      validator: validator.isEmail,
      message: "Please provide a valid email address",
    },
  },
  domain: {
    type: String,
  },
  googleId: {
    type: String,
    unique: true,
    sparse: true,
  },
  linkedInId: {
    type: String,
    unique: true,
    sparse: true,
  },
  authMethods: [
    {
      type: {
        type: String,
        enum: ["google", "email", "linkedin"],
      },
      verified: { type: Boolean, default: false },
    },
  ],

  gender: {
    type: String,
  },
  password: {
    type: String,
    required: function () {
      return !(this.googleId || this.linkedInId);
    },
    minlength: [6, "Password must be at least 6 characters long"],
  },
  type: {
    type: String,
    enum: ["college Student", "fresher"],
  },
  otp: { type: String },
  otpExpiry: { type: Date },
  avatar: { type: String },
  resume: {
    file: { type: String },
    parsedText: { type: String },
  },
  isResumeParsed: { type: Boolean, default: false, required: true },
  profile: {
    hollrAiCalls: [
      {
        type: mongoose.Schema.Types.ObjectId,
        ref: "HollrAiCall",
      }
    ]
,    
    socials: {
      linkedin: {
        type: String,
        default: "",
      },
      github: {
        type: String,
        default: "",
      },
    },

    phoneNumber: {
      type: String,
      unique: true,
      validate: {
        validator: (value) => validator.isMobilePhone(value, "any"),
        message: "Please provide a valid phone number",
      },
    },
    age: {
      type: Number,
      min: [0, "Age must be a positive number"],
    },
    education: {
      institutionName: { type: String }, // School or college name
      institutionType: {
        type: String,
        enum: ["school", "college"],
        required: false,
      }, // Type of institution
      major: { type: String },
      cgpa: { type: Number }, // Optional
      startYear: {
        type: Number,
        required: false,
        validate: {
          validator: Number.isInteger,
          message: "Start year must be an integer",
        },
      }, // Start year
      endYear: {
        type: Number,
        required: false,
        validate: {
          validator: Number.isInteger,
          message: "End year must be an integer",
        },
      }, // End year
      year: {
        type: Number,
      }, // Current year
    },
    experience: [
      {
        company: { type: String },
        position: { type: String },
        startYear: { type: Number },
        endYear: { type: Number },
      },
    ],
    projects: [
      {
        projectName: { type: String },
        description: { type: String },
        status: {
          type: String,
          enum: ["completed", "inprogress"],
          required: false,
        },
        demoLink: { type: String },
        startYear: { type: Number, required: false },
        endYear: { type: Number, required: false   },
      },
    ],
    skills: [
      {
        skillName: { type: String, required: false },
        proficiency: { type: String, required: false },
      },
    ],
    certifications: [
      {
        certificationName: { type: String, required: false },
        organization: { type: String, required: false },
        issueDate: { type: Date, required: false },
      },
    ],
    positionsOfResponsibility: [
      {
        position: { type: String, required: false },
        organization: { type: String, required: false },
        issueDate: { type: Date, required: false },
      },
    ],
    languages: [
      {
        language: { type: String, required: false },
        proficiency: { type: String, required: false },
      },
    ],
    tags: [
      {
        type: String,
        enum: [
          "it",
          "marketing",
          "finance and accounting",
          "sales",
          "human resources",
          "legal",
          "retail",
          "customer service",
        ],
        required: true,
      },
    ],
    seeking: {
      type: String,
      enum: ["jobs", "internship"],
    },
    personalityBlueprint: [
      {
        questionId: {
          type: mongoose.Schema.Types.ObjectId,
          ref: "Question",
          required: true,
        },
        selectedOption: { type: Number, required: true },
      },
    ],
    traitScores : {
      o:{
        type: Number, 
        // required: true
      },
      c:{
        type: Number, 
        // required: true
      },
      e:{
        type: Number, 
        // required: true
      },
      a:{
        type: Number, 
        // required: true
      },
      n:{
        type: Number, 
        // required: true
      }
    },
    mbtiType: {
      type: String
    }
  },
  credits: {
    type: Number,
    default: 1000,
    required: true,
  },

  appliedJobs: [
    {
      jobId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "Job",
        required: true,
      },
      appliedAt: { type: Date, default: Date.now },
      status: {
        type: String,
        enum: ["in-progress", "matched", "closed"],
        default: "in-progress", // Default status when the user applies
      },
      compatibilityScore: {
        type: Number,
      },
      superLike: {
        promoted : Boolean,
        videoNote: {
          type: String,
          default: "",
        },
      },
    },
  ],
  savedJobs: [
    {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Job",
    },
  ],
  isTutorialWatched: { type: Boolean, default: false },
  swipes: { type: Number, default: 0 },
  lastSwipeDate: { type: Date },
  createdAt: { type: Date, default: Date.now, required: true },
  updatedAt: { type: Date, default: Date.now, required: true },
});

ApplicantSchema.methods.recordSwipe = async function () {
  const today = new Date();
  today.setHours(0, 0, 0, 0); // Start of today

  // Check if swipes need to reset
  if (!this.lastSwipeDate || this.lastSwipeDate < today) {
    this.swipes = 1; // Reset swipes to 1
    this.lastSwipeDate = today; // Set last swipe date to today
  } else {
    this.swipes += 1; // Increment swipe count for today
  }

  // Save the changes to the document
  await this.save();

  return this; // Return the updated instance
};

ApplicantSchema.statics.canSwipeToday = async function (applicantId) {
  const applicant = await this.findById(applicantId);

  if (!applicant) {
    return { allowed: false, dailySwipes: 0 };
  }

  const today = new Date();
  today.setHours(0, 0, 0, 0);

  if (!applicant.lastSwipeDate || applicant.lastSwipeDate < today) {
    // If it's a new day, reset the swipe count
    return { allowed: true, dailySwipes: 0 };
  }

  // Check if swipes are within limit
  const dailySwipes = applicant.swipes || 0;
  return { allowed: dailySwipes < 5, dailySwipes };
};

ApplicantSchema.pre("save", async function (next) {
  this.updatedAt = new Date();
  if (!this.isModified("password")) return next();
  this.password = await bcrypt.hash(this.password, 10);
  next();
});

//Indexes
ApplicantSchema.index({ email: 1 }, { unique: true }); // Email lookup
ApplicantSchema.index({ googleId: 1 }); // Google OAuth
ApplicantSchema.index({ "appliedJobs.jobId": 1, "appliedJobs.status": 1 }); // Job applications filter
ApplicantSchema.index({ savedJobs: 1 }); // Saved jobs query

module.exports = mongoose.model("Applicant", ApplicantSchema);
