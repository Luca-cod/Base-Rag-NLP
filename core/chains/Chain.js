"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.filterByVisualizationType = void 0;
exports.createRagChain = createRagChain;
var prompts_1 = require("@langchain/core/prompts");
var runnables_1 = require("@langchain/core/runnables");
var output_parsers_1 = require("@langchain/core/output_parsers");
// Example filter for visualizationType
exports.filterByVisualizationType = {
    'BOXIO': 'controller',
    'WS558': 'smart_light',
    'SMABIT_AV2010_32': 'thermostat',
    'LED_DRIVER': 'smart_light',
    'EASTRON_SDM630': 'energy_meter',
    'GEWISS_GWA1531': 'actuator',
    'VAYYAR_CARE': 'sensor'
};
function createRagChain(llm, vectorStore, preFilteredDocs) {
    return __awaiter(this, void 0, void 0, function () {
        var docs, context, prompt, chain;
        var _this = this;
        return __generator(this, function (_a) {
            if (!llm || typeof llm.invoke !== 'function') {
                throw new Error("Invalid or uninitialized LLM.");
            }
            if (!vectorStore || typeof vectorStore.asRetriever !== 'function') {
                throw new Error("Invalid VectorStore: missing asRetriever.");
            }
            if (preFilteredDocs && preFilteredDocs.length > 0) {
                docs = preFilteredDocs;
                console.log("Using pre-filtered documents from runRAGSystem");
                console.log("Documents after filtering:", docs.length);
            }
            else {
                throw new Error("No documents provided to createRagChain");
            }
            context = docs.map(function (doc) { return doc.pageContent; }).join("\n\n");
            prompt = prompts_1.ChatPromptTemplate.fromTemplate("\n\nYou are an assistant specialized in the analysis of home automation systems. Your task is to provide accurate information based solely on the data provided in the context.\nIt is essential that you maintain a strictly analytical approach and never add information that is not explicitly present in the data available to you.\n\nIMPORTANT! \n\nYou MUST respond with VALID JSON only. No additional text, no explanations.\n\nCORE OPERATING PRINCIPLES\n\nYour primary responsibility is to be a faithful interpreter of the provided data. \nThis means you must always base your answers on the concrete facts present in the context, avoiding any form of speculation,\n logical deduction, or addition of information based on general knowledge. \nIf a piece of information is not explicitly present in the data, it is your responsibility to clearly communicate this to the user.\n\nBut if the context is very verbose, use the data that is most relevant to you for that particular query, For example, if the query requires only one parameter, return that parameter without returning all the others.\n\n\n\u26A0\uFE0F USE EXCLUSIVELY THE DATA PROVIDED IN THE GIVEN CONTEXT\n\nYou may INTERPRET clearly labeled parameters (e.g., \"setpoint\" means target temperature) \nbut NEVER infer values not present in the data.\n\n\u26A0\uFE0F ALWAYS STATE WHEN INFORMATION IS NOT AVAILABLE IN THE DATA\nAVAILABLE DEVICES CONTEXT\n\n{context}\nUSER REQUEST\n\n{query}\nANALYSIS METHODOLOGY\nPreliminary Verification Phase\n\nBefore formulating any answer, you must conduct a systematic analysis of the available data. Start by identifying whether the requested device or information is actually present in the provided context. Never assume a device exists just because the user mentions it\u2014always verify its presence in the data.\n\nCarefully examine the structure of the data to understand which parameters are actually documented and which values are specified. Remember that the absence of a parameter in the data does not mean it doesn't exist in the real system, but simply that you do not have sufficient information to discuss it.\nInformation Extraction Process\n\nOnce the requested device is verified to be present, proceed with the information extraction following a rigorous methodology. Focus exclusively on what is literally present in the data, using the exact field names and specified values.\n\nFrom the context provided to you, only use the devices requested by the query!\nIf the query required the all parameters of devices, give me all technical details of every parameter. If the query required some parameters, show pnly the paramters reuired.\n\nFor general device information, look for and report:\n\n    The name of the device as specified in the relevant field\n\n    The unique identifier (UUID), if available\n\n    The device category\n\n    The specific display type or model\n\n    Any numerical identifiers\n\nFor technical parameters, for each item present in the context, document:\n\n    The exact name of the parameter\n\n    The data type (DataType) with its precise numerical value\n\n    The current or default value, if specified\n\n    The unit of measurement, when available\n\n    The logging type (LogType) and notification frequency\n\n    Any supported operations (switch, button, update, etc.)\n\n    Minimum and maximum values, if defined\n\nCommunication of Limitations\n\nIt is essential that you always communicate the limitations of the information at your disposal. Users must understand that your analysis is based on the specific data provided and that additional information not visible in the current context may exist.\n\n\nThe context contains COMPLETE DEVICES with their parameters. Each device has:\n\n1. A PARENT CHUNK (ID: 0) - Contains device overview information\n2. PARAMETER CHUNKS (ID: 1, 2, 3...) - Individual parameters of the device\n\nDEVICE-PARAMETER RELATIONSHIPS:\n- Parent chunk UUID: {{parent_uuid}} \u2192 Parameter chunks with parentUuid: {{parent_uuid}}\n- Seq Session: {{session_id}} \u2192 All chunks with same session belong to same device\n\nEXAMPLE STRUCTURE OF Hierarchy:\n\u2022 Smart Light Controller (Parent - ID: 0)\n  - brightness (Parameter - ID: 1) \n  - power_consumption (Parameter - ID: 2)\n  - status (Parameter - ID: 3)\n\n IMPORTANT!: For specific queries requiring chunks with chunkType = 'detail', use only the core parameters you consider most important.\n\n  ANALYSIS RULES:\n  1. NEVER list individual parameters as separate devices\n  2. ALWAYS group parameters under their parent device\n  3. If a parameter chunk is found, find its parent device first\n  4. Ignore parameter chunks that don't belong to any parent\n\nRESPONSE FORMAT (JSON ONLY, NO EXTRA TEXT):\n\nExample for query specific, includes parameters:\n{{\n  \"devices\": [\n    {{\n      \"name\": \"Device Name\",\n      \"uuid\": \"device-uuid\",\n      \"category\": \"Category Name\",\n      \"parameters\": [\n        {{\"name\": \"param1\", \"value\": \"X\"}},\n        {{\"name\": \"param2\", \"value\": \"Y\"}}\n      ]\n  }}\n  ]\n  }}  \n\n  5. **COMPLETE DETAIL QUERIES** (like \"show all details\", \"all parameters\"):\n   {{\n     \"devices\": [\n       {{\n         \"name\": \"Device Name\",\n         \"uuid\": \"device-uuid\", \n         \"category\": \"Category Name\",\n         \"visualizationType\": \"Type\",\n         \"parameters\": [\n           {{\"name\": \"param1\", \"value\": \"X\", \"unit\": \"unit\"}},\n           {{\"name\": \"param2\", \"value\": \"Y\", \"unit\": \"unit\"}}\n         ]\n       }}\n     ]\n   }}\n\nExample of Generic Query like  **UUID-ONLY QUERIES** (like \"show me UUIDs\", \"list UUIDs\"):\n   {{\n     \"uuids\": [\n       \"uuid-1\",\n       \"uuid-2\", \n       \"uuid-3\"\n     ]\n   }}\n\n   3. **DEVICE LIST QUERIES** (like \"list devices\", \"show me devices\"):\n   {{\n     \"devices\": [\n       {{\n         \"name\": \"Device Name\",\n         \"uuid\": \"device-uuid\",\n         \"category\": \"Category Name\"\n       }}\n     ]\n   }}\n\n\nOrganize your response in a clear and professional manner, always beginning with a precise identification of the device being analyzed.\n\nDevice Identification:\nPresent the exact name of the device, its full UUID, category, and display type, using only the values found in the data.\n\nParameter Analysis:\nFor each parameter identified in the context, provide a complete description that includes all available technical details. Do not merely list the parameters\u2014explain the characteristics of each one based on the structured data.\n\nStatement of Limitations:\nAlways conclude by specifying that your analysis is based on the currently available data in the context and that additional parameters or configurations may exist but are not visible at this time.\nHandling Absence Cases\n\nIf the requested device is not present in the context, communicate this immediately and provide a list of the devices actually available in the data. Never attempt to provide information about devices that are not present or to make assumptions about their existence.\nEXAMPLES OF APPROPRIATE COMMUNICATION\n\n\n\nFINAL VERIFICATION\n\nBefore providing your response, always conduct a systematic check:\n\n    Is every piece of information you mentioned literally present in the context?\n\n    Did you avoid adding personal interpretations or logical deductions?\n\n    Did you clearly communicate the limitations of the available information?\n\n    Is your response based exclusively on the provided data?\n\nOnly after positively confirming all these points should you proceed with formulating the final answer.\n\nResponse:\n    ");
            /*
          
          const prompt = ChatPromptTemplate.fromTemplate(`
          
              HOME AUTOMATION DATA EXTRACTOR
              Respond with VALID JSON only. No additional text.
              
              CONTEXT:
              {context}
              
              QUERY:
              {query}
              
              CORE PRINCIPLES:
              - Use ONLY data explicitly present in context
              - NEVER invent, infer, or modify values
              - Adapt response structure to query type
              - Group parameters under parent devices
              
              RESPONSE SCHEMAS (choose based on query):
              
              1. UUID-ONLY (queries with "UUID", "uuids", "identifiers"):
              {{ "uuids": ["uuid1", "uuid2"] }}
              
              2. DEVICE LIST (queries with "list", "show devices", "what devices"):
              {{ "devices": [
                {{"name": "Device1", "uuid": "uuid1", "category": "Category"}}
              ]}}
              
              3. BASIC INFO (queries about device names/categories):
              {{ "devices": [
                {{"name": "Device1", "uuid": "uuid1", "category": "Category", "visualizationType": "Type"}}
              ]}}
              
              4. PARAMETER-SPECIFIC (queries mentioning specific parameters):
              {{ "devices": [
                {{"name": "Device1", "uuid": "uuid1", "parameters": [
                  {{"name": "param1", "value": "exact_value", "unit": "unit"}}
                ]}}
              ]}}
              
              5. COMPLETE DETAILS (queries with "all details", "full parameters"):
              {{ "devices": [
                {{"name": "Device1", "uuid": "uuid1", "category": "Category",
                 "visualizationType": "Type", "parameters": [
                  {{"name": "param1", "value": "value1", "unit": "unit1", "dataType": "type1"}},
                  {{"name": "param2", "value": "value2", "unit": "unit2", "dataType": "type2"}}
                ]}}
              ]}}
              
              CONTEXT ADAPTATION:
              - If context has FEW parameters (summary chunks) → Return basic device info
              - If context has MANY parameters (detail chunks) → Return only relevant parameters
              - If query is generic → Return appropriate level of detail
              - If query is specific → Return only requested information
              
              QUERY ANALYSIS FOR: "{query}"
              - This appears to be a □ UUID-only □ Device list □ Basic info □ Parameter-specific □ Complete details request
              - Use schema: □1 □2 □3 □4 □5
              - Include parameters: □ No □ Only requested □ All available
              
              DATA VALIDATION:
              - Verify all UUIDs exist in context
              - Use exact values from context
              - Omit missing information
              - Group parameters under correct devices
              
              Response:
              `);
          */
            //const chunks = [];
            console.log(" Documents received from AdaptiveRecovery:", docs.map(function (d) { return ({
                name: d.metadata.name,
                uuid: d.metadata.uuid,
                category: d.metadata.category,
                chunkType: d.metadata.chunkType
            }); }));
            chain = runnables_1.RunnableSequence.from([
                runnables_1.RunnablePassthrough.assign({
                    context: function () { return context; },
                    query: function (input) {
                        if (!input.query || typeof input.query !== 'string') {
                            throw new Error("Invalid Query!");
                        }
                        return input.query;
                    },
                    originalDocs: function () { return docs; }
                }),
                prompt,
                llm,
                {
                    func: function (response, input) { return __awaiter(_this, void 0, void 0, function () {
                        var textResponse, parser, parsedResponse, parseError_1, cleanedResponse, error_1;
                        return __generator(this, function (_a) {
                            switch (_a.label) {
                                case 0:
                                    _a.trys.push([0, 5, , 6]);
                                    textResponse = void 0;
                                    if (typeof response === 'string') {
                                        textResponse = response;
                                    }
                                    else if (response === null || response === void 0 ? void 0 : response.content) {
                                        textResponse = response.content;
                                    }
                                    else if (response === null || response === void 0 ? void 0 : response.text) {
                                        textResponse = response.text;
                                    }
                                    else {
                                        textResponse = JSON.stringify(response);
                                    }
                                    //Clean the response of any extra characters
                                    textResponse = textResponse.trim();
                                    // If it starts with ```json, remove the backticks
                                    if (textResponse.startsWith('```json')) {
                                        textResponse = textResponse.replace(/^```json\s*/, '').replace(/\s*```$/, '');
                                    }
                                    else if (textResponse.startsWith('```')) {
                                        textResponse = textResponse.replace(/^```\s*/, '').replace(/\s*```$/, '');
                                    }
                                    parser = new output_parsers_1.JsonOutputParser();
                                    parsedResponse = void 0;
                                    output_parsers_1.JsonOutputParser;
                                    _a.label = 1;
                                case 1:
                                    _a.trys.push([1, 3, , 4]);
                                    return [4 /*yield*/, parser.parse(textResponse)];
                                case 2:
                                    parsedResponse = _a.sent();
                                    /*  const valid = validation(parsedResponse);
                                      if (!valid) {
                                        console.error("Validation from JsonSchema failed!", validation.errors);
                                      } else {
                                        console.log(" JSON parsing successful");
                                      }*/
                                    console.log(" JSON parsing successful");
                                    return [3 /*break*/, 4];
                                case 3:
                                    parseError_1 = _a.sent();
                                    console.warn(" JSON parsing failed, attempting manual fix");
                                    Error;
                                    //Manual correction of the JSON
                                    try {
                                        cleanedResponse = textResponse
                                            .replace(/^[^{]*({.*})[^}]*$/, '$1') // Extracts only the JSON object
                                            .replace(/,(\s*[}\]])/g, '$1');
                                        parsedResponse = JSON.parse(cleanedResponse);
                                        console.log(" Manual JSON fix successful");
                                    }
                                    catch (manualError) {
                                        console.error(" Both JSON parsing methods failed:", manualError);
                                        // As a fallback, create a valid JSON structure
                                        parsedResponse = {
                                            deviceAnalysis: {
                                                deviceFound: false,
                                                deviceInfo: {},
                                                parameters: [],
                                                limitations: "Error parsing LLM response - response format was invalid"
                                            },
                                            summary: "Unable to parse the response from the language model",
                                            rawResponse: textResponse
                                        };
                                    }
                                    return [3 /*break*/, 4];
                                case 4: 
                                // Return the parsed JSON as a string
                                return [2 /*return*/, JSON.stringify(parsedResponse, null, 2)];
                                case 5:
                                    error_1 = _a.sent();
                                    console.error(" Error in response processing:", error_1);
                                    return [3 /*break*/, 6];
                                case 6: return [2 /*return*/];
                            }
                        });
                    }); }
                }
            ]);
            return [2 /*return*/, chain];
        });
    });
}
function formatFallbackSection(fallbacks) {
    var parts = [" ADDITIONAL INFORMATION:"];
    fallbacks.forEach(function (doc) {
        parts.push("\u2022 ".concat(doc.metadata.name || 'Information chunk'));
    });
    return parts.join('\n');
}
function getCategoryName(category) {
    var categories = {
        0: 'Controller',
        1: 'Gateway',
        2: 'Bridge',
        11: 'Actuator',
        12: 'Smart Switch',
        15: 'Smart Light',
        18: 'Sensor',
        // Aggiungi altre categorie se necessario
    };
    return categories[category] || "Category ".concat(category);
}
function getDataTypeName(typeCode) {
    var types = {
        0: 'number',
        1: 'decimal',
        2: 'boolean',
        3: 'string',
        4: 'enumeration',
        5: 'integer'
    };
    return types[typeCode] || 'unknown';
}
