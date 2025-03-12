const mongoose = require('mongoose');

const CallMessageSchema = new mongoose.Schema({
    callSid: { type: String, required: true, index: true },
    role: { type: String, enum: ["user", "assistant"], required: true },
    content: { type: String, required: true },
    timestamp: { type: Date, default: Date.now }
  });

module.exports= mongoose.model("CallMessage", CallMessageSchema);