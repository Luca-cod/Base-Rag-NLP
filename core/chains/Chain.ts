

import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { Ollama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import { EndpointMetadata } from "../retrieval/loaders/loadDocumentJSON3 copy.js";
import { Document as LangChainDocument } from "langchain/document";
import { ExtendsSeqMetadata } from "../retrieval/splitters/SecondSplit2.js"
import { JsonOutputParser } from "@langchain/core/output_parsers";


// Example filter for visualizationType
export const filterByVisualizationType = {
  'BOXIO': 'controller',
  'WS558': 'smart_light',
  'SMABIT_AV2010_32': 'thermostat',
  'LED_DRIVER': 'smart_light',
  'EASTRON_SDM630': 'energy_meter',
  'GEWISS_GWA1531': 'actuator',
  'VAYYAR_CARE': 'sensor'
}


export async function createRagChain(llm: Ollama, vectorStore: FaissStore, preFilteredDocs?: LangChainDocument[]): Promise<RunnableSequence> {




  if (!llm || typeof llm.invoke !== 'function') {
    throw new Error("Invalid or uninitialized LLM.");
  }


  if (!vectorStore || typeof vectorStore.asRetriever !== 'function') {
    throw new Error("Invalid VectorStore: missing asRetriever.");
  }


  let docs: LangChainDocument[];


  if (preFilteredDocs && preFilteredDocs.length > 0) {
    docs = preFilteredDocs;
    console.log("Using pre-filtered documents from runRAGSystem");
    console.log("Documents after filtering:", docs.length);
  } else {
    throw new Error("No documents provided to createRagChain");
  }


  /*docs.forEach((doc, i) => {
    console.log(`[${i}] Type: ${doc.metadata.type}, ChunkType: ${doc.metadata.chunkType}, Name: ${doc.metadata.name}`);
  });*/



  /*  } else {

  // Fallback: fai la ricerca come prima
  console.log(" No pre-filtered documents, I search in the vectorStore");
  //const dynamicFilter = createDynamicFilter(query);
  const fallbackFilter = {
    "metadata.type": "installation-config",
    "metadata.hasAreaInfo": true // Cerca chunk che hanno info di area, anche se sono detail
  };


  //Da cancellare

  docs = await vectorStore
    .asRetriever({
      filter: fallbackFilter
    })
    .invoke(query);

  console.log("Documents retrieved with chunkType:",
    docs.map(d => d.metadata.chunkType).join(', ')
  );
}

// Remove duplicates (same uuid and chunkType)
const seenKeys = new Set();
docs = docs.filter(doc => {
  const key = `${doc.metadata?.uuid}-${doc.metadata?.chunkType}`;
  if (seenKeys.has(key)) return false;
  seenKeys.add(key);

  return true;
});
console.log("Controlliamo quanti documenti ha tolto questo filtro:", seenKeys.size);
*/


  /**ATTENTION: The context uses a hierarchical numbering system:
- Chunks with ID "0", "1", etc. are PARENT DEVICES
- Chunks with ID "0.1", "0.2", etc. are PARAMETERS of PARENT DEVICES
- Always group parameters under their parent device */


  const context = docs.map(doc => doc.pageContent).join("\n\n");


  //PROMPT WITH ANTI-HALLUCINATIONS!!
  const prompt = ChatPromptTemplate.fromTemplate(`

You are an assistant specialized in the analysis of home automation systems. Your task is to provide accurate information based solely on the data provided in the context.
It is essential that you maintain a strictly analytical approach and never add information that is not explicitly present in the data available to you.

IMPORTANT! 

You MUST respond with VALID JSON only. No additional text, no explanations.

CORE OPERATING PRINCIPLES

Your primary responsibility is to be a faithful interpreter of the provided data. 
This means you must always base your answers on the concrete facts present in the context, avoiding any form of speculation,
 logical deduction, or addition of information based on general knowledge. 
If a piece of information is not explicitly present in the data, it is your responsibility to clearly communicate this to the user.

But if the context is very verbose, use the data that is most relevant to you for that particular query, For example, if the query requires only one parameter, return that parameter without returning all the others.


⚠️ USE EXCLUSIVELY THE DATA PROVIDED IN THE GIVEN CONTEXT

You may INTERPRET clearly labeled parameters (e.g., "setpoint" means target temperature) 
but NEVER infer values not present in the data.

⚠️ ALWAYS STATE WHEN INFORMATION IS NOT AVAILABLE IN THE DATA
AVAILABLE DEVICES CONTEXT

{context}
USER REQUEST

{query}
ANALYSIS METHODOLOGY
Preliminary Verification Phase

Before formulating any answer, you must conduct a systematic analysis of the available data. Start by identifying whether the requested device or information is actually present in the provided context. Never assume a device exists just because the user mentions it—always verify its presence in the data.

Carefully examine the structure of the data to understand which parameters are actually documented and which values are specified. Remember that the absence of a parameter in the data does not mean it doesn't exist in the real system, but simply that you do not have sufficient information to discuss it.
Information Extraction Process

Once the requested device is verified to be present, proceed with the information extraction following a rigorous methodology. Focus exclusively on what is literally present in the data, using the exact field names and specified values.

From the context provided to you, only use the devices requested by the query!
If the query required the all parameters of devices, give me all technical details of every parameter. If the query required some parameters, show pnly the paramters reuired.

For general device information, look for and report:

    The name of the device as specified in the relevant field

    The unique identifier (UUID), if available

    The device category

    The specific display type or model

    Any numerical identifiers

For technical parameters, for each item present in the context, document:

    The exact name of the parameter

    The data type (DataType) with its precise numerical value

    The current or default value, if specified

    The unit of measurement, when available

    The logging type (LogType) and notification frequency

    Any supported operations (switch, button, update, etc.)

    Minimum and maximum values, if defined

Communication of Limitations

It is essential that you always communicate the limitations of the information at your disposal. Users must understand that your analysis is based on the specific data provided and that additional information not visible in the current context may exist.


The context contains COMPLETE DEVICES with their parameters. Each device has:

1. A PARENT CHUNK (ID: 0) - Contains device overview information
2. PARAMETER CHUNKS (ID: 1, 2, 3...) - Individual parameters of the device

DEVICE-PARAMETER RELATIONSHIPS:
- Parent chunk UUID: {{parent_uuid}} → Parameter chunks with parentUuid: {{parent_uuid}}
- Seq Session: {{session_id}} → All chunks with same session belong to same device

EXAMPLE STRUCTURE OF Hierarchy:
• Smart Light Controller (Parent - ID: 0)
  - brightness (Parameter - ID: 1) 
  - power_consumption (Parameter - ID: 2)
  - status (Parameter - ID: 3)

 IMPORTANT!: For specific queries requiring chunks with chunkType = 'detail', use only the core parameters you consider most important.

  ANALYSIS RULES:
  1. NEVER list individual parameters as separate devices
  2. ALWAYS group parameters under their parent device
  3. If a parameter chunk is found, find its parent device first
  4. Ignore parameter chunks that don't belong to any parent

RESPONSE FORMAT (JSON ONLY, NO EXTRA TEXT):

Example for query specific, includes parameters:
{{
  "devices": [
    {{
      "name": "Device Name",
      "uuid": "device-uuid",
      "category": "Category Name",
      "parameters": [
        {{"name": "param1", "value": "X"}},
        {{"name": "param2", "value": "Y"}}
      ]
  }}
  ]
  }}  

  5. **COMPLETE DETAIL QUERIES** (like "show all details", "all parameters"):
   {{
     "devices": [
       {{
         "name": "Device Name",
         "uuid": "device-uuid", 
         "category": "Category Name",
         "visualizationType": "Type",
         "parameters": [
           {{"name": "param1", "value": "X", "unit": "unit"}},
           {{"name": "param2", "value": "Y", "unit": "unit"}}
         ]
       }}
     ]
   }}

Example of Generic Query like  **UUID-ONLY QUERIES** (like "show me UUIDs", "list UUIDs"):
   {{
     "uuids": [
       "uuid-1",
       "uuid-2", 
       "uuid-3"
     ]
   }}

   3. **DEVICE LIST QUERIES** (like "list devices", "show me devices"):
   {{
     "devices": [
       {{
         "name": "Device Name",
         "uuid": "device-uuid",
         "category": "Category Name"
       }}
     ]
   }}


Organize your response in a clear and professional manner, always beginning with a precise identification of the device being analyzed.

Device Identification:
Present the exact name of the device, its full UUID, category, and display type, using only the values found in the data.

Parameter Analysis:
For each parameter identified in the context, provide a complete description that includes all available technical details. Do not merely list the parameters—explain the characteristics of each one based on the structured data.

Statement of Limitations:
Always conclude by specifying that your analysis is based on the currently available data in the context and that additional parameters or configurations may exist but are not visible at this time.
Handling Absence Cases

If the requested device is not present in the context, communicate this immediately and provide a list of the devices actually available in the data. Never attempt to provide information about devices that are not present or to make assumptions about their existence.
EXAMPLES OF APPROPRIATE COMMUNICATION



FINAL VERIFICATION

Before providing your response, always conduct a systematic check:

    Is every piece of information you mentioned literally present in the context?

    Did you avoid adding personal interpretations or logical deductions?

    Did you clearly communicate the limitations of the available information?

    Is your response based exclusively on the provided data?

Only after positively confirming all these points should you proceed with formulating the final answer.

Response:
    `);

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


  console.log(" Documents received from AdaptiveRecovery:", docs.map(d => ({
    name: d.metadata.name,
    uuid: d.metadata.uuid,
    category: d.metadata.category,
    chunkType: d.metadata.chunkType
  })));

  /*  //  let JsonSchema = await ApplyJsonSchema(); //Obj
    console.log(typeof loadSchema);
    console.log("COsa c'era in loadSchema?");
    let schema = loadSchema(directoryPath);
    if (typeof schema === 'string') {
      schema = JSON.parse(schema);
    } else {
      console.log("Di che tipo è schema? ", typeof schema)
    }
    const ajv = new Ajv();
    //JsonSchema = typeof JsonSchema === 'string' ? JSON.parse(JsonSchema) : JsonSchema;
    const validation = ajv.compile(schema);*/

  //console.log("Contesto prepared:", context.substring(0, 100) + "...");
  // console.log("Contesto prepared:", context.substring(0, 10000));
  const chain = RunnableSequence.from([
    RunnablePassthrough.assign({
      context: () => context,
      query: (input: { query: string }) => {
        if (!input.query || typeof input.query !== 'string') {
          throw new Error("Invalid Query!");
        }

        return input.query;
      },
      originalDocs: () => docs

    }),
    prompt,
    llm,
    {

      func: async (response: any, input: any) => {


        // Extract text content from response
        // Estrai il contenuto testuale dalla risposta
        /*const textResponse = typeof response === 'string'
          ? response
          : response?.content || response?.text || JSON.stringify(response);*/


        try {
          // Extract text content from response
          let textResponse: string;

          if (typeof response === 'string') {
            textResponse = response;
          } else if (response?.content) {
            textResponse = response.content;
          } else if (response?.text) {
            textResponse = response.text;
          } else {
            textResponse = JSON.stringify(response);
          }

          //Clean the response of any extra characters
          textResponse = textResponse.trim();

          // If it starts with ```json, remove the backticks
          if (textResponse.startsWith('```json')) {
            textResponse = textResponse.replace(/^```json\s*/, '').replace(/\s*```$/, '');
          } else if (textResponse.startsWith('```')) {
            textResponse = textResponse.replace(/^```\s*/, '').replace(/\s*```$/, '');
          }

          // parsing 
          const parser = new JsonOutputParser();
          let parsedResponse;
          JsonOutputParser
          try {
            parsedResponse = await parser.parse(textResponse);
            /*  const valid = validation(parsedResponse);
              if (!valid) {
                console.error("Validation from JsonSchema failed!", validation.errors);
              } else {
                console.log(" JSON parsing successful");
              }*/

            console.log(" JSON parsing successful");
          } catch (parseError) {
            console.warn(" JSON parsing failed, attempting manual fix");
            Error

            //Manual correction of the JSON
            try {
              // Remove invalid characters and attempt parsing
              const cleanedResponse = textResponse
                .replace(/^[^{]*({.*})[^}]*$/, '$1') // Extracts only the JSON object
                .replace(/,(\s*[}\]])/g, '$1'); // Remove commas before closing clauses

              parsedResponse = JSON.parse(cleanedResponse);
              console.log(" Manual JSON fix successful");
            } catch (manualError) {
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
          }

          // Return the parsed JSON as a string
          return JSON.stringify(parsedResponse, null, 2);
        } catch (error) {
          console.error(" Error in response processing:", error);

        }
      }
    }
  ]);

  return chain;
}



function formatFallbackSection(fallbacks: LangChainDocument[]): string {
  const parts = [" ADDITIONAL INFORMATION:"];

  fallbacks.forEach(doc => {
    parts.push(`• ${doc.metadata.name || 'Information chunk'}`);
  });

  return parts.join('\n');
}


function getCategoryName(category: number): string {
  const categories: Record<number, string> = {
    0: 'Controller',
    1: 'Gateway',
    2: 'Bridge',
    11: 'Actuator',
    12: 'Smart Switch',
    15: 'Smart Light',
    18: 'Sensor',
    // Aggiungi altre categorie se necessario
  };
  return categories[category] || `Category ${category}`;
}

function getDataTypeName(typeCode: number): string {
  const types: Record<number, string> = {
    0: 'number',
    1: 'decimal',
    2: 'boolean',
    3: 'string',
    4: 'enumeration',
    5: 'integer'
  };
  return types[typeCode] || 'unknown';
}
