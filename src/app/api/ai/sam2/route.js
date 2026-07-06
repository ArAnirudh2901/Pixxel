// Legacy alias — the SAM route was renamed to /api/ai/sam3 (it has served
// SAM 3.1 prompts since the masking-service split). Kept so older clients keep working.
export { POST, maxDuration } from '../sam3/route'
