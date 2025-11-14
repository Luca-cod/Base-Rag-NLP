import { ExtendDocument, targetFile, filePath } from "../../../config/RAGNLP.js";
import { promises as fs } from "fs";
import { Document } from "langchain/document"; // Document is not an array and does not have a push() method
import { buildGlobalPartitionMap } from "./buildGlobalPartitionsMap.js";

/*

This function takes a complex JSON configuration of an installation, validates it, builds maps between areas,
partitions, and endpoints, and generates a single document ready to be chunked or indexed, including all relevant
statistics and metadata.

It creates a single document that contains:

    • All processed endpoints in an array  
    • All processed areas in an array  
    • Complete installation statistics  
    • A global partition map  


    
----    Detailed breakdown     ----

1- Load and validate the file

Reads the JSON manually (via fs.readFile).  
Checks that it’s not empty and that parsing is successful.  
Verifies the presence of key elements (endpoints, areas, etc.).

2- Builds relationships and maps

    Uses functions like buildGlobalPartitionMap, buildAreaPartitionMaps, buildEndpointAreaRelations.  
    These define:

        – which endpoints belong to which areas,  
        – which partitions are shared,  
        – how many areas and devices exist, etc.  

    Essentially, it reconstructs the logical topology of the system.

3- Calculates global statistics

    Counts sensors, actuators, and controllers.  
    Counts partitions and areas.  
    Retrieves installation version (major, minor, revision).

4- Creates a semantic “summary” document

    A single LangChain `Document` whose `pageContent` looks like this:  
        {
            "type": "installation-config",
            "statistics": { ... },
            "endpoints": [...],
            "areas": [...],
            "globalPartitionMap": {...}
        }

    And — most importantly — a very rich metadata object:
        metadata: {
            type: 'installation-config',
            chunkType: 'summary',
            totalEndpoints: 732,
            totalAreas: 12,
            hasPartitions: true,
            major: 2,
            minor: 5,
            revision: "2025.11",
            ...
        }

    These metadata fields are later used during:

    – Vector search (for filtering or weighting documents)  
    – The answer generation phase (to provide targeted context)

5- Handles fallback

    If something fails (empty file, parsing error, etc.), it generates a fallback document (“system unavailable”).
*/



export type DocumentType = 'installation-config';
export type ChunkType = "summary" | "detail" | "area" | "fallback";


export interface EndpointMetadata {
    // Basic metadata
    source: string;            // Source file name
    loc: string;               // Full file path
    type: DocumentType;        // Document type
    isValid: boolean;
    timestamp: string;
    chunkType: ChunkType;      // Chunk type (summary, detail, area, fallback)

    // REQUIRED endpoint data
    uuid?: string;             // Device UUID
    name?: string;             // Device name
    category?: number;         // Category (0, 15, 11, 18)
    visualizationType?: string; // Visualization type (BOXIO, VAYYAR_CARE, etc.)

    // OPTIONAL data
    categoryName?: string;
    partitions?: string[];
    location?: string[];
    areaNames?: string[];
    areaUuids?: string[];
    id?: string;
    parametersCount?: number;
    defaultParameter?: string;

    // ========================================
    // METADATA FOR TWO-STAGE CHUNKING
    // ========================================
    isPrimaryChunk?: boolean;
    chunkStrategy?: 'two_stage' | 'standard'; // Identifies which chunking strategy was used

    parameterStartIndex?: number;
    parameterEndIndex?: number;
    totalParameters?: number;

    visualizationCategory?: string;

    deviceType: string;
    globalPartitionMapArray?: Array<[string, string]>;

    // Filtering flags
    hasAreaInfo?: boolean;
    hasEndpoints?: boolean;
    hasConfiguration?: boolean;
    hasControlParams?: boolean;
    isFirstFloor?: string;
    isSecondFloor?: string;

    // Area chunk data
    areaName?: string;
    areaUuid?: string;
    areaIndex?: number;
    devicesCount?: number;
    floorName?: string;
    deviceTypes?: string[];
    deviceCategories?: number[];
    partitionNames?: string[];
    partitionIds?: string[];

    // Detail chunk data
    isSensor?: boolean;
    isActuator?: boolean;
    isController?: boolean;
    hasMeasurementParameters?: boolean;
    hasEnumerationParameters?: boolean;
    hasConfigParameters?: boolean;
    parameterUnits?: string[];
    parameterDataTypes?: string[];

    // Splitting info
    subChunkIndex?: number;
    totalSubChunks?: number;
    splitField?: string;
    fullUuid?: string;
    isSubChunk?: boolean;
    warning?: string;
    error?: string;

    parameterNames?: string[];
    parameterOperations?: string[];
    hasMeasurementParams?: boolean;

    totalEndpoints?: number;
    totalAreas?: number;
    hasPartitions?: boolean; // Indicates whether the document has partitions
    installationName?: string;
    revision?: string;
    minor?: number;
    major?: number;

    [key: string]: any;

    sequenceNumberMetadata?: SeqMetadata; // Calculated during splitting — cannot be defined yet at chunk creation
}

export interface SeqMetadata {
    sessionId: string;
    chunkId: number;
    totalChunks: number;
    parentChunkId?: number;
    isParent?: boolean;
    isAckChunk?: boolean;
}

export interface Parameter {
    name: string;
    dataType: number;
    unit?: string;
    operation?: { type: string };
    logType?: number;
    defaultStateValue?: string;
    notifyFrequency?: number;
    maxVal: number[];
    minVal: number[];
    [key: string]: any;
}

// INTERFACES TO IMPROVE MAPPING CLARITY
export interface AreaPartitionMap {
    areaUuid: string;
    areaName: string;
    partitions: Array<{
        uuid: string;
        name: string;
    }>;
}

export interface EndpointAreaRelation {
    endpointUuid: string;
    endpointName: string;
    areaUuid: string;
    areaName: string;
    partitionUuids: string[];
    location: string[];
}

export interface LoadDocumentResult {
    documents: ExtendDocument[];
    partitionMap: Map<string, string>;
}



async function loadDocumentsJSON(): Promise<LoadDocumentResult> {
    const processedUUIDs = new Set<string>();
    let rawContent: string;
    const documents: ExtendDocument[] = [];

    try {
        rawContent = await fs.readFile(filePath, 'utf-8');
    } catch (error) {
        console.error("ERROR: Document not found or not readable!");
        throw new Error("Execution blocked: file doesn't exist");
    }

    if (!rawContent || rawContent.trim().length === 0) {
        console.error("ERROR: Empty document!");
        throw new Error("Execution blocked: file contents empty.");
    }

    let jsonContent: any;
    try {
        jsonContent = JSON.parse(rawContent);
    } catch (parseError) {
        console.error("ERROR: JSON parsing failed", parseError);
        throw new Error("Invalid JSON format in configuration");
    }

    if (!jsonContent || typeof jsonContent !== 'object') {
        throw new Error("Invalid JSON structure: root must be an object");
    }


    const hasValidEndpoints = Array.isArray(jsonContent.endpoints) && jsonContent.endpoints.length > 0;
    const hasValidAreas = Array.isArray(jsonContent.areas) && jsonContent.areas.length > 0;

    if (!hasValidEndpoints) {
        console.warn("No valid endpoint found in JSON content");
        return getFallbackDocument(new Error("No valid endpoints in configuration"));
    }

    console.log(`Data structure: ${jsonContent.endpoints.length} endpoints, ${jsonContent.areas?.length || 0} areas`);

    // Build global maps
    const globalPartitionMap = buildGlobalPartitionMap(jsonContent);
    const areaPartitionMaps = hasValidAreas ? buildAreaPartitionMaps(jsonContent) : [];
    const endpointAreaRelations = hasValidAreas
        ? buildEndpointAreaRelations(jsonContent, areaPartitionMaps)
        : new Map<string, EndpointAreaRelation>();


    // Create installation content
    const installationContent = {
        type: "installation-config",
        metadata: jsonContent.metadata || {},
        statistics: {
            totalEndpoints: jsonContent.endpoints?.length || 0,
            totalAreas: jsonContent.areas?.length || 0,
            totalPartitions: globalPartitionMap.size,
            sensorCount: jsonContent.endpoints.filter((ep: any) => ep.category === 18).length || 0,
            actuatorCount: jsonContent.endpoints?.filter((ep: any) => [11, 12, 15].includes(ep.category)).length || 0,
            controllerCount: jsonContent.endpoints?.filter((ep: any) => [0, 1, 2].includes(ep.category)).length || 0
        },
        endpoints: jsonContent.endpoints,
        areas: jsonContent.areas,
        globalPartitionMap: Object.fromEntries(globalPartitionMap)
    };

    const mainDocument = new Document({
        pageContent: JSON.stringify(installationContent),
        metadata: {
            source: targetFile,
            loc: filePath,
            type: 'installation-config',
            isValid: true,
            timestamp: new Date().toISOString(),
            name: jsonContent.metadata?.name,
            chunkType: 'summary',

            installationName: jsonContent.metadata?.name || 'installation-config',
            revision: jsonContent.metadata?.revision,
            deviceType: 'installation',
            totalEndpoints: jsonContent.endpoints?.length || 0,
            totalAreas: jsonContent.areas?.length || 0,
            hasPartitions: globalPartitionMap.size > 0,
            hasAreaInfo: hasValidAreas,
            major: jsonContent.metadata?.major,
            minor: jsonContent.metadata?.minor,
        }
    }) as unknown as ExtendDocument;
    documents.push(mainDocument);

    console.log("Single raw document created for two-stage chunking");
    return {
        documents,
        partitionMap: globalPartitionMap
    };
}

/**
 * ========================================================================================================
 *                                             MAPPING FUNCTIONS
 * ========================================================================================================
 */

/**
 * Fundamental function for:
 * - Creating the mapping between areas and partitions
 * - Handling both object and UUID-string partition formats
 * - Providing robust validation
 * - Resolving partition names reliably
 */
export function buildAreaPartitionMaps(jsonContent: any): AreaPartitionMap[] {
    const maps: AreaPartitionMap[] = [];

    if (!jsonContent.areas || !Array.isArray(jsonContent.areas)) {
        console.warn("No areas found in JSON file");
        return maps;
    }

    for (const [index, area] of jsonContent.areas.entries()) {
        try {
            // Strict validation for area objects
            if (!area || typeof area !== 'object') {
                console.warn(`Area ${index} invalid:`, area);
                continue;
            }

            if (!area.uuid || !area.name) {
                console.warn(`Area ${index} missing UUID or name:`, {
                    uuid: area.uuid,
                    name: area.name
                });
                continue;
            }

            const areaMap: AreaPartitionMap = {
                areaUuid: area.uuid,
                areaName: area.name,
                partitions: []
            };

            // Secure partition handling
            if (Array.isArray(area.partitions)) {
                for (const [partIndex, partition] of area.partitions.entries()) {
                    if (!partition) {
                        console.warn(` Partition ${partIndex} in area ${area.name} is null/undefined`);
                        continue;
                    }

                    // Handle both objects and UUID strings
                    const partitionUuid = typeof partition === 'string' ? partition : partition.uuid;
                    const partitionName =
                        typeof partition === 'string'
                            ? `Partition_${partition.substring(0, 8)}`
                            : partition.name;

                    if (partitionUuid && partitionName) {
                        areaMap.partitions.push({ uuid: partitionUuid, name: partitionName });
                    } else {
                        console.warn(` Partition ${partIndex} in area ${area.name} has incomplete data`);
                    }
                }
            }

            maps.push(areaMap);
            console.log(`Mapped Area: ${area.name} (${areaMap.partitions.length} partitions)`);
        } catch (error) {
            console.error(`Error processing area ${index}:`, error);
            continue;
        }
    }
    console.log(`Area maps created: ${maps.length}`);
    return maps;
}


// Iterates through endpoints and finds areas by shared partitions
function buildEndpointAreaRelations(
    jsonContent: any,
    areaPartitionMaps: AreaPartitionMap[]
): Map<string, EndpointAreaRelation> {
    const relations = new Map<string, EndpointAreaRelation>();

    console.log("Building endpoint-area relationships...");

    // Input validation
    if (!jsonContent?.areas || !Array.isArray(jsonContent.areas)) {
        console.warn("No areas found in JSON to build relationships");
        return relations;
    }

    if (!Array.isArray(areaPartitionMaps) || areaPartitionMaps.length === 0) {
        console.warn("No partition maps available to build relationships");
        return relations;
    }

    console.log(`Areas to process: ${jsonContent.areas.length}, Partition maps: ${areaPartitionMaps.length}`);
    console.log("Processed areas:", JSON.stringify(jsonContent.areas), "Partition names:", JSON.stringify(areaPartitionMaps));

    let totalEndpointsProcessed = 0;
    let totalRelationsCreated = 0;

    for (const [endpointIndex, endpoint] of jsonContent.endpoints.entries()) {
        try {
            totalEndpointsProcessed++;

            if (!endpoint || !endpoint.uuid) {
                console.warn(`Endpoint ${endpointIndex} invalid or missing UUID`);
                continue;
            }

            if (!Array.isArray(endpoint.partitions) || endpoint.partitions.length === 0) {
                continue;
            }

            const endpointName = endpoint.name || `Device_${endpoint.uuid.substring(0, 8)}`;

            for (const areaMap of areaPartitionMaps) {
                const sharedPartitions = areaMap.partitions.filter(areaPartition =>
                    endpoint.partitions.includes(areaPartition.uuid)
                );

                if (sharedPartitions.length > 0) {
                    const relation: EndpointAreaRelation = {
                        endpointUuid: endpoint.uuid,
                        endpointName,
                        areaUuid: areaMap.areaUuid,
                        areaName: areaMap.areaName,
                        partitionUuids: sharedPartitions.map(p => p.uuid),
                        location: sharedPartitions.map(p => p.name)
                    };

                    if (relations.has(endpoint.uuid)) {
                        const existing = relations.get(endpoint.uuid);
                        console.log(`Endpoint ${endpoint.uuid} already mapped to ${existing?.areaName}, also found in ${areaMap.areaName}`);
                    } else {
                        relations.set(endpoint.uuid, relation);
                        totalRelationsCreated++;
                        console.log(`Relationship created: ${endpointName} -> ${areaMap.areaName} (${sharedPartitions.length} shared partitions)`);
                    }
                }
            }
        } catch (endpointError) {
            console.error(`Error processing endpoint ${endpointIndex}:`, endpointError);
            continue;
        }
    }

    // Final report
    console.log("\nENDPOINT-AREA RELATIONSHIP REPORT:");
    console.log(`Endpoints processed: ${totalEndpointsProcessed}`);
    console.log(`Relationships created: ${totalRelationsCreated}`);
    console.log(`Unique relationships in map: ${relations.size}`);

    // Detailed debug
    if (relations.size > 0) {
        console.log("\nFirst 3 relationships created:");
        let count = 0;
        for (const [uuid, relation] of relations.entries()) {
            if (count >= 3) break;
            console.log(`   ${count + 1}. ${relation.endpointName} -> ${relation.areaName}`);
            console.log(`      UUID: ${uuid}`);
            console.log(`      Shared partitions: ${relation.location.join(', ')}`);
            count++;
        }
    } else {
        console.warn("\nWARNING: No relationships created!");
        console.log("Debug: Verify that areas and endpoints share partitions");

        console.log("Partitions by area:");
        areaPartitionMaps.forEach(areas => {
            console.log(`   ${areas.areaName}: [${areas.partitions.map(p => p.name).join(', ')}]`);
        });

        console.log("Partitions by endpoint (first 5):");
        jsonContent.endpoints.slice(0, 5).forEach((ep: any) => {
            if (ep.partitions?.length > 0) {
                console.log(`   ${ep.name}: [${ep.partitions.join(', ')}]`);
            }
        });
    }

    return relations;
}

function getFallbackDocument(error: any): LoadDocumentResult {
    const fallbackUUID = 'fallback-' + Math.random().toString(36).substring(2, 9);
    const fallBack = {
        pageContent: JSON.stringify({
            error: "Failed to load document",
            message: error instanceof Error ? error.message : String(error),
            fallbackType: "empty_system"
        }),
        metadata: {
            source: 'fallback',
            loc: 'internal',
            type: 'installation-config' as const, // fallback
            isValid: false,
            timestamp: new Date().toISOString(),
            uuid: fallbackUUID,
            name: 'Fallback Document',
            category: -1,
            visualizationType: 'N/A',
            deviceType: 'other',
            categoryName: 'fallback',
            visualizationCategory: 'fallback',
            id: '0',
            partitions: [],
            location: [],
            areaNames: [],
            areaUuids: [],
            parametersCount: 0,
            defaultParameter: '',
            chunkStrategy: 'standard' as const,
            chunkType: 'fallback' as const,
            hasAreaInfo: true
        },
        readableText: "System temporarily unavailable. Please check the configuration and try again."
    };
    return {
        documents: [fallBack],
        partitionMap: new Map()
    };
}
