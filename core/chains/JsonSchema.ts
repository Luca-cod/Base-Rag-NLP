import { promises as fs } from "fs";
import { directoryPath } from "src/config/RAG.js";
import path from 'path';




const fallBackPrompt = `{{
    "metadata": {{
      "id": "string",
      "boxioId": "string",
      "name": "string",
      "revision": "string",
      "major": number,
      "minor": number,
      "clusterId": number
    }},
    "areas": [
      {{
        "uuid": "string",
        "id": number,
        "name": "string",
        "partitions": [
          {{
            "uuid": "string",
            "id": number,
            "name": "string"
          }}
        ],
        "longitude": number,
        "latitude": number
      }}
    ],
    "endpoints": [
      {{
        "uuid": "string",
        "id": number,
        "category": number,
        "name": "string",
        "defaultParameter": "string",
        "parameters": [
          {{
            "name": "string",
            "dataType": number,
            "unit": "string",
            "logType": number,
            "operation": {{
              "type": "string"
            }}
          }}
        ],
        "partitions": ["string"],
        "visualizationType": "string"
      }}
    ],
    "statistics": {{
      "totalEndpoints": number,
      "totalParameters": number,
      "sensorCount": number
    }}
    }}`
  ;
// ========================================== CARICO IL JSON-SCHEMA ============================================0
export async function loadSchema(directoryPath: string): Promise<any> {


  const schemaPath = path.join(directoryPath, 'schema.json');

  try {
    await fs.access(schemaPath);
    console.log(`Schema found in: ${schemaPath}`); // Debug del percorso
    const content = await fs.readFile(schemaPath, 'utf-8');
    console.log("JSON schema found in:", schemaPath);

    //Parsing del file
    const jsonObject = JSON.parse(content);
    //Rendo una stringa leggibile il file oggetto json parsato

    return jsonObject;
    //return JSON.parse(content);
  } catch (error) {
    console.error("Error loading schema:", {
      error: error instanceof Error ? error.message : String(error),
      path: schemaPath
    });
    throw new Error(`Unable to load JSON schema from path: ${schemaPath}`);
  }
}


/*              FUNZIONE CHE VALIDA LA RISPOSTA DELL'LLM
async function validateLLMResponse(response: any, directoryPath: string): Promise<string> {

//Load the schema using your existing function
const schema = await loadSchema(directoryPath);


try {

if (typeof response === 'string') {
  return response;

}

let responseText = typeof response === 'string'
  ? response
  : response?.content || response?.text || JSON.stringify(response);

//Try to parse as JSON
let parsed

try {
  parsed = JSON.parse(responseText);
} catch (e) {
  console.warn("Response is not valid JSON, keep as text");
  return responseText;
}

//Validation with JSON Schema
const validationResult = validator.validate(parsed, schema);

if (validationResult.valid) {
  console.log("Response validated successfully against JSON Schema");
  return JSON.stringify(parsed, null, 2);
} else {
  console.warn("Response validation failed", validationResult.errors);

  const errorMessages = validationResult.errors.map((error: any) =>
    `${error.property}: ${error.message}`
  );

  return JSON.stringify({
    error: "Invalid automation structure",
    validationErrors: errorMessages,
    rawResponse: parsed
  }, null, 2);
}

    // Validazione struttura base
    if (parsed.response && typeof parsed.response === 'string') {
      console.log(" Validate text response");
      return JSON.stringify(parsed);
    }
 
    // Validazione per liste di dispositivi
    if (parsed.devices && Array.isArray(parsed.devices)) {
      // Filtra dispositivi validi
      const validDevices = parsed.devices.filter((d: any) => {
        const hasName = d.name && typeof d.name === 'string';
        if (!hasName) {
          console.warn(`Unnamed devices: ${JSON.stringify(d)}`);
        }
        return hasName;
      });
 
      parsed.devices = validDevices;
 
      if (validDevices.length === 0) {
        parsed.response = "Nessun dispositivo valido trovato per la query";
        delete parsed.devices;
      }
 
      console.log(`✅ ${validDevices.length} dispositivi validati`);
    }
 
    return JSON.stringify(parsed, null, 2);
             
} catch (e) {
console.warn("⚠️ Validazione fallita, ritorno risposta grezza:", e);
return typeof response === 'string' ? response : JSON.stringify(response);
}
}
*/


//========================== Function of loading and interpretation of a example file =============================

async function loadExample(directoryPath: string): Promise<string> {

  const schemaPath = path.join(directoryPath, 'example-03.json');

  try {
    await fs.access(schemaPath);
    console.log(`Schema found in: ${schemaPath}`); // Debug del percorso
    const content = await fs.readFile(schemaPath, 'utf-8');
    //Controllo validità JSON
    const jsonObject = JSON.parse(content);//se fallisce JSON non valido
    const formatted = JSON.stringify(jsonObject, null, 2); // Indentazione a 2 spazi
    return formatted;
  } catch (error) {
    console.error("Error loading schema:", {
      error: error instanceof Error ? error.message : String(error),
      path: schemaPath
    });
    throw new Error(`Unable to load JSON schema from path: ${schemaPath}`);
  }
}

export async function ApplyJsonSchema() {
  try {
    const schema = await loadSchema(directoryPath);

    //Adding the Schema to Prompt:
    const schemaPrompt = `You are an assistant specialized in the analysis of home automation systems. Your task is to provide accurate information based solely on the data provided in the context.
It is essential that you maintain a strictly analytical approach and never add information that is not explicitly present in the data available to you.

IMPORTANT! 
The format response should be in JSON!
You MUST respond with VALID JSON only. No additional text, no explanations.

CORE OPERATING PRINCIPLES

Your primary responsibility is to be a faithful interpreter of the provided data. This means you must always base your answers on the concrete facts present in the context, avoiding any form of speculation, logical deduction, or addition of information based on general knowledge. 
If a piece of information is not explicitly present in the data, it is your responsibility to clearly communicate this to the user.



⚠️ USE EXCLUSIVELY THE DATA PROVIDED IN THE GIVEN CONTEXT
⚠️ DO NOT ADD INFORMATION BASED ON GENERAL KNOWLEDGE OR DEDUCTIONS

You may only deduce information if:
- A parameter is clearly labeled (e.g. "setpoint") and there is no ambiguity in meaning
- The context includes only one value for that parameter

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
If the query required the all parameters of devices, show all parameters. If the query required some parameters, show pnly the paramters reuired.

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

EXAMPLE STRUCTURE:
• Smart Light Controller (Parent - ID: 0)
  - brightness (Parameter - ID: 1) 
  - power_consumption (Parameter - ID: 2)
  - status (Parameter - ID: 3)

  ANALYSIS RULES:
  1. NEVER list individual parameters as separate devices
  2. ALWAYS group parameters under their parent device
  3. If a parameter chunk is found, find its parent device first
  4. Ignore parameter chunks that don't belong to any parent

RESPONSE STRUCTURE:
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

Correct Phrasing:

    "Based on the provided data, the parameter has the following characteristics..."

    "The analyzed context includes the following configured elements..."

    "The available data show that this parameter has a specific DataType..."

    "I do not have sufficient information in the current context to describe this aspect..."

Phrasing to Strictly Avoid:

    Never mention creation dates, specific temperatures, or monitoring frequencies unless explicitly present

    Never describe "typical" behaviors or "standard" values of devices

    Never state the operational status of a device without explicit data

    Never add technical details based on general product knowledge

FINAL VERIFICATION

Before providing your response, always conduct a systematic check:

    Is every piece of information you mentioned literally present in the context?

    Did you avoid adding personal interpretations or logical deductions?

    Did you clearly communicate the limitations of the available information?

    Is your response based exclusively on the provided data?

Only after positively confirming all these points should you proceed with formulating the final answer.

Response: \n\nJSON SCHEMA: ${schema}`;

    return schemaPrompt;
  } catch (e) {
    console.warn(` Schema loading failed, using basic JSON structure `);
    return fallBackPrompt;
  }
}