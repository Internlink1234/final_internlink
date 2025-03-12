const mongoose = require('mongoose');


const HollrAiCallSchema = new mongoose.Schema({
  callSid: { type: String, required: true, unique: true },
  name: { type: String, required: false }, // Not present in new payload, kept optional for backward compatibility
  email: {type: String},
  fromNumber: { type: String, required: true },
  toNumber: { type: String, required: true },
  direction: { type: String, enum: ["outbound-api", "inbound"], required: true },
  status: { type: String, enum: ["completed", "failed", "in-progress"], required: true },
  duration: { type: Number, required: true },
  aiCallSummary: { type: String },
  callSummary: { type: String },
  confidenceScore: { type: Number },
  assistantAccuracy: { type: Number },
  userAccuracy: { type: Number },
  assistantClarity: { type: Number },
  userClarity: { type: Number },
  messages: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "CallMessage",
  },
  source : {type : String, enum : ["callbackpage","inbound-call","outbound-call"]},
  createdAt: { type: Date, default: Date.now }
});


module.exports = HollrAiCall = mongoose.model("HollrAiCall", HollrAiCallSchema);