import { GoogleGenerativeAI } from "@google/generative-ai";

// Validate API key
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
    throw new Error("GOOGLE_API_KEY environment variable is not set");
}

export async function queryLLM(query: string, relevantData: string[]) {
    console.log('🤖 Initializing Gemini...');
    try {
        const genAI = new GoogleGenerativeAI(GEMINI_API_KEY || "");
        // Using pro model instead of flash for better stability
        const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

        console.log('📝 Preparing prompt with context...');
        const prompt = `Based on the following context, answer the question:

Context: ${relevantData.join("\n")}

Question: ${query}

Answer:`;

        console.log('🔄 Generating response...');
        const result = await model.generateContent(prompt);
        if (!result.response) {
            throw new Error("No response received from Gemini");
        }
        console.log('✅ Response generated successfully');
        return result.response.text();
    } catch (error: any) {
        console.error("❌ Error querying Gemini:", {
            message: error.message,
            status: error.status,
            details: error.errorDetails || 'No additional details'
        });
        if (error.status === 403) {
            throw new Error("Invalid or missing API key. Please check your GOOGLE_API_KEY environment variable.");
        }
        throw error;
    }
}