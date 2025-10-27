import { AutomationConfig } from "./RAG.js";

export function parserLLMResponse(raw: string | object): AutomationConfig | string | null {
    if (typeof raw === 'string') {
        try {
            return JSON.parse(raw);
        } catch (e) {
            console.error("Parsing fallito", e);
            return null;
        }

    }
    return raw as AutomationConfig | string | null;
}