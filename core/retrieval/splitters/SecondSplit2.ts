

import { Endpoint } from "src/config/RAG.js";
import { ChunkType, EndpointMetadata, SeqMetadata } from "../loaders/loadDocumentJSON.js";




export interface ExtendsSeqMetadata extends EndpointMetadata {
    sequenceNumberSystem?: SeqMetadata;
    parentChunkId?: string;
    chunkId?: string;
    isParent?: boolean;
    totalChildren?: number;

}

export type Chunk = {
    pageContent: string;
    metadata: Record<string, any>;
};

/**il problema corretto. Il RecursiveCharacterTextSplitter 
 * fallisce perché installation-config.json è un unico oggetto JSON compatto senza separatori naturali 
 * che il text splitter possa usare. 
 * 
 * Il nuovo approccio strutturale risolve il problema
Perché falliva prima:

RecursiveCharacterTextSplitter cerca separatori di testo (\n\n, ., spazi)
installation-config.json è un blob JSON compatto senza separatori naturali
Il sistema ciclava sempre sullo stesso oggetto di 37.005 caratteri

Come funziona la nuova soluzione:

Parsing della struttura JSON: Identifica arrays principali (area, partitions, endpoints, devices)
Split per elementi: Ogni elemento dell'array diventa un chunk separato
Fallback a proprietà: Se non trova arrays, divide per proprietà principali dell'oggetto
Validazione garantita: Controlla che ogni chunk sia ≤ chunkSize, altrimenti applica text splitting*/
export function splitLargeJsonObjectByArrayField(
    obj: any, //rappresenta il dispositivo/area completo, dati originali
    maxChunkSize: number,
    // metadata: EndpointMetadata,
    chunk: Chunk,
    depth: number = 0 //perr evitare loop infiniti
): Chunk[] {
    let count = 0;
    const metadata = chunk.metadata as EndpointMetadata;
    console.log(` Splitting object of ${JSON.stringify(obj).length} chars`);
    console.log(`   Original metadata: ${JSON.stringify({
        chunkType: metadata.maxChunkSize.metadataType,
        source: metadata.source,
        uuid: metadata.metadata.uuid,
        name: metadata.metadata.name,
        cateogy: metadata.metadata.category,
        isFirstFloor: metadata.metadata.isFirstFloor,
        isSecondFloor: metadata.metadata.isSecondFloor,
        floorLocation: metadata.metadata.floorLocation,
        partitionNames: metadata.metadata.partitionNames,
        visualizationType: metadata.metadata.visualizationType,
        category: metadata.metadata.category
    })}`);



    console.log(`🔍 DEBUG splitLargeJsonObjectByArrayField:`);
    console.log(`   - Input object keys: ${Object.keys(obj).join(', ')}`);
    console.log(`   - Metadata: ${JSON.stringify({
        name: metadata.name,
        chunkType: metadata.chunkType,
        source: metadata.source
    })}`);

    // Verifica se sta ricevendo la struttura corretta
    if (obj.data) {
        console.log(`   - Data object keys: ${Object.keys(obj.data).join(', ')}`);
    }
    if (obj.endpoints) {
        console.log(`   - Endpoints count: ${obj.endpoints.length}`);
    }





    // Protezione ricorsione
    if (depth > 5) {
        console.warn("Max depth reached, truncating");
        return [{
            pageContent: JSON.stringify(obj).substring(0, maxChunkSize),
            metadata: {
                ...metadata,
                warning: "Max depth reached - truncated",
                error: "Structural splitting failed"
            }
        }
        ];
    }


    try {

        const normalizeMetadata = normalizelocationMetadata(metadata);

        // PRESERVA TUTTI I METADATI CRITICI
        const criticalMetadata: EndpointMetadata = {
            /*  isFirstFloor: metadata.isFirstFloor || 'Unknown',
              isSecondFloor: metadata.isSecondFloor || 'Unknown',
              floorLocation: metadata.floorLocation || 'unknown',
              partitionNames: metadata.partitionNames || [],
              hasAreaInfo: metadata.hasAreaInfo || false,*/
            ...normalizeMetadata,
            splitAttempted: true,
            visualizationType: metadata.visualizationType,
            category: metadata.category
        };


        // Gestione specifica per la struttura dei chunk installation-config
        // const chunkType = obj.chunkType || metadata.chunkType;

        // CERCA ARRAY IN obj.data PRIMA CHE IN obj
        let targetObject = obj;  //targetObject rappresenta i dati specifici, lo usiamo per lo splitting degli array
        if (obj.parameterData && typeof obj.parameterData === 'object') {
            console.log(`   - Switching to parameterData for array search`);
            targetObject = obj.parameterData;
        }
        else if (obj.data && typeof obj.data === 'object') {
            targetObject = obj.data;
            console.log(" Switching to data object for array search");
        }

        console.log(`   - Final target object ${Object.keys(targetObject).join(', ')}`);


        const topLevelArrays = ['area', 'partitions', 'endpoints', 'devices', 'parameters'];

        //DEBUG for visula what arrays is found
        console.log(`   -Checking for arrays in targetObject`);
        topLevelArrays.forEach(field => {
            if (Array.isArray(targetObject[field])) {
                console.log(`  FOUND: ${field} with ${targetObject[field].length}`);
            }
        });

        //Versione con Sequence number per collegare children a father
        const splitSessionId = `split-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
        const totalChunks = calculateTotalChunks(targetObject, topLevelArrays); //Function to implement

        //Create a father chunk
        const parentChunk = createParentChunk(obj, criticalMetadata, splitSessionId, totalChunks);
        const chunks: Chunk[] = [parentChunk];


        //create chunk father (header)
        let chunkIndex = 1; //Inizia da 1 (0) il padre




        // Cerca campi array da splittare - ora cerchiamo direttamente nell'oggetto, non in obj.data
        for (const field of topLevelArrays) {
            const array = targetObject[field]; // Cambiato da obj.data[field] a obj[field]

            if (Array.isArray(array) && array.length > 0) {
                // const baseData = { ...targetObject };
                console.log(`Splitting field ${field} with ${array.length} items`);



                for (const [index, item] of array.entries()) {

                    const chunkMetadata: EndpointMetadata = {  //For more informations 
                        ...criticalMetadata,
                        chunkType: 'detail' as const,

                        //System SequenceNumb like IP
                        sequenceNumberSystem: {
                            sessionId: splitSessionId,
                            chunkId: chunkIndex,
                            totalChunks: totalChunks,
                            parentChunkId: 0, //0 = chunk padre
                            isSequneceNumbChunk: true,
                            role: chunkIndex === 0 ? 'parent_device' : 'device_parameter',
                            relationship: chunkIndex === 0 ? 'contains_parameters' : `parameter_of_device_${metadata.uuid}`
                        },
                        chunkId: `0.${index + 1}`, // Sistema gerarchico 0.1, 0.2, 0.3
                        parentChunkId: "0", //Explicit reference to father
                        structuralChunk: true,
                        sourceArray: field,
                        arrayIndex: index,
                        totalArrayItems: array.length,
                        subChunkIndex: count++,
                        uuid: item.uuid || item.id || `${field}-${index}`,
                        name: item.name || `${field} ${index + 1}`,
                        /*
                        
                        / UUID unico per ogni chunk figlio (evita deduplicazione)
                        uuid: `${criticalMetadata.uuid}-param-${index}`,
  
                         // ⚠️ NOME del device padre (per query testuali)*/
                        fatherName: criticalMetadata.name,
                        isSubChunk: true,
                        usbChunkIndex: index,  //Indice unico
                        totalSubChunks: array.length, // Totale chunk
                        splitField: field, // Campo splittato
                        uniqueChunkId: `${metadata.uuid}-params-${index}-${field}`, // ID univoco

                        //Link to device father, hierarchic relation
                        parentUuid: criticalMetadata.uuid,
                        parentName: criticalMetadata.name,
                        deviceCategory: criticalMetadata.category,
                        visualizationType: criticalMetadata.visualizationType,
                        hierarchicalRole: chunkIndex === 0 ? 'device_parent' : 'parameter_child',
                        parentDeviceName: metadata.name, // Nome leggibile, non solo UUID
                        parentDeviceCategory: metadata.category,

                        // Metadati specifici del parametro
                        isParameterChunk: true,
                        parameterIndex: index,
                        parameterName: item.name,
                        parameterDataType: item.dataType,

                        // ⚠️ Eredita tutti i flag di query del parent
                        hasControlParams: criticalMetadata.hasControlParams,
                        hasMeasurementParams: criticalMetadata.hasMeasurementParams,
                        hasConfigParams: criticalMetadata.hasConfigParams,
                        hasAutomationParams: criticalMetadata.hasAutomationParams,

                        // Location metadata ereditata
                        isFirstFloor: criticalMetadata.isFirstFloor,
                        isSecondFloor: criticalMetadata.isSecondFloor,
                        floorLocation: criticalMetadata.floorLocation,
                        partitionNames: criticalMetadata.partitionNames,


                    };



                    const childChunk = createChildChunk(item, chunkMetadata, field, index);
                    if (childChunk.pageContent.length <= maxChunkSize) {
                        chunks.push(childChunk);
                    } else {
                        console.log(`🔁 Recursing into oversized ${field} item ${index}`);

                        const subChunk = splitLargeJsonObjectByArrayField(
                            JSON.parse(childChunk.pageContent),
                            maxChunkSize,
                            //Creo oggetto Chunk che unisce anche oggetto EndpointMetadata
                            {
                                pageContent: childChunk.pageContent,
                                metadata: chunkMetadata
                            },
                            depth + 1
                        );
                        chunks.push(...subChunk);


                    }

                    chunkIndex++;

                    //      STAMPA CHUNKS CHILD
                    console.log(` This is a child ${JSON.stringify(childChunk)}- ${metadata.ackMetadata?.sessionId}  of ---> ${JSON.stringify(parentChunk)} with 
                    ${JSON.stringify(chunkIndex)}`);

                }
                console.log(`   Split completed for ${field}. Chunks created: ${chunks.length}`);

            }



            /*   const str = JSON.stringify(chunkObj);
 
               if (str.length <= maxChunkSize) {
                   chunks.push({
                       pageContent: str,
                       metadata: chunkMetadata // Cambiato: metadata diretto, non annidato
                   });
               } else {
                   //console.warn(`Element too large even on its own in '${field}':`, item); ------> stampa una miriade di roba
 
                   // Se anche il singolo elemento è troppo grande, fallback
                   // Ricorsione per elementi troppo grandi
                   console.log(`🔁 Recursing into oversized ${field} item ${index}`);
                   const subChunks = splitLargeJsonObjectByArrayField(
                       chunkObj,
                       maxChunkSize,
                       chunkMetadata, //Dovrei modificarlo in chunckMetadata ma mi da errori di incompatibilità
                       depth + 1
                   );
                   chunks.push(...subChunks);
 
               }
           }
 
           //  DEBUG: Verifica che i metadata di location siano preservati
           console.log(`   Split completed for ${field}. Chunks created: ${chunks.length}`);
           if (chunks.length > 0) { 
               const firstChunk = chunks[0].metadata;
               console.log(`   First chunk location metadata:`, {
                   isFirstFloor: firstChunk.isFirstFloor,
                   isSecondFloor: firstChunk.isSecondFloor,
                   floorLocation: firstChunk.floorLocation,
                   partitionNames: firstChunk.partitionNames
               });
 
               // VALIDAZIONE: Assicurati che floorLocation non sia undefined
               if (!firstChunk.floorLocation || firstChunk.floorLocation === 'undefined') {
                   console.warn(`⚠️ FloorLocation is ${firstChunk.floorLocation} for chunk ${firstChunk.name}`);
               }
         }


          //Return father + children
             console.log(`✅ Created hierarchical chunks: 1 parent + ${chunks.length} children`);
          return [parentChunk, ...chunks];



          // returns after the first splittable field found
           // return chunks;
             }*/
        }
        if (chunks.length > 0) {
            return chunks;
        }
        // FALLBACK: Se non trova array, splitta per proprietà principali
        console.log(" No arrays found, attempting property-based split");
        return splitByProperties(obj, maxChunkSize, criticalMetadata, depth);

    } catch (error) {
        console.error(" Error in splitLargeJsonObjectByArrayField:", error);

        let errorMessages = "Uknown error";
        if (error instanceof Error) errorMessages = error.message;

        return [{

            pageContent: JSON.stringify(obj).substring(0, maxChunkSize),
            metadata: {
                ...metadata,
                error: `Splitting error: ${errorMessages}`
            }
        }];
    }


}


function createParentChunk(originalObj: any, baseMetadata: EndpointMetadata, sessionId: string, totalChunks: number): Chunk {
    const parentContent = {
        type: 'parent',
        ackHeader: {
            sessionId: sessionId,
            chunkId: 0, // ID 0 = chunk padre
            totalChunks: totalChunks,
            message: `This device has ${totalChunks - 1} parameter chunks. Refer to chunks 1-${totalChunks - 1} for details.`
        },
        deviceInfo: {
            name: baseMetadata.name,
            uuid: baseMetadata.uuid,
            category: baseMetadata.category
        }
    };

    return {
        pageContent: JSON.stringify(parentContent),
        metadata: {
            ...baseMetadata,
            chunkType: 'summary' as const,
            sequenceNumberSystem: {
                sessionId: sessionId,
                chunkId: 0,
                totalChunks: totalChunks,
                isParent: true
            }
        }
    };
}

function createChildChunk(item: any, metadata: EndpointMetadata, field: string, index: number): Chunk {
    const chunkObj = {
        type: 'parameter',
        ackInfo: {
            sessionId: metadata.sequenceNumberSystem!.sessionId,
            chunkId: metadata.sequenceNumberSystem!.chunkId,
            parentChunkId: 0,
            parameterIndex: index
        },
        parameterData: item
    };

    //Preseva che critical metadatas of parent
    const preservedMetadata: EndpointMetadata = {
        ...metadata,
        visualizationType: metadata.visualizationType,
        category: metadata.category,
        categoryName: metadata.categoryName,
        deviceType: metadata.deviceType,
        name: metadata.name,
        uuid: metadata.uuid,

        // Location metadata
        isFirstFloor: metadata.isFirstFloor,
        isSecondFloor: metadata.isSecondFloor,
        floorLocation: metadata.floorLocation,
        partitionNames: metadata.partitionNames,

        // Flags di query
        hasControlParams: metadata.hasControlParams,
        hasMeasurementParams: metadata.hasMeasurementParams,
        hasConfigParams: metadata.hasConfigParams
    };

    return {
        pageContent: JSON.stringify(chunkObj),
        metadata: preservedMetadata
    };
}

function calculateTotalChunks(targetObject: any, fields: string[]): number {
    let total = 1; // Sempre almeno il chunk padre

    for (const field of fields) {
        const array = targetObject[field];
        if (Array.isArray(array)) {
            total += array.length;
        }
    }

    return total;
}


// FUNZIONE DI FALLBACK PER SPLITTARE PER PROPRIETÀ
function splitByProperties(obj: any, maxChunkSize: number, metadata: EndpointMetadata, depth: number): Chunk[] {
    const chunks: Chunk[] = [];
    const properties = ['area', 'endpoints', 'metadata', 'configurations'];

    for (const prop of properties) {
        if (obj[prop] && typeof obj[prop] === 'object') {
            const propChunk = {
                type: prop,
                [prop]: obj[prop]
            };

            const str = JSON.stringify(propChunk);
            if (str.length <= maxChunkSize) {
                chunks.push({
                    pageContent: str,
                    metadata: {
                        ...metadata,
                        chunkType: 'partial' as ChunkType,
                        splitProperty: prop,
                        floorLocation: metadata.floorLocation || 'Unknown'
                    }
                }
                );
            }
        }
    }

    if (chunks.length > 0) {
        console.log(`✅ Fallback split created ${chunks.length} chunks with location metadata`);
        return chunks;
    }

    return [{
        pageContent: JSON.stringify(obj).substring(0, maxChunkSize),
        metadata: {
            ...metadata,
            warning: "Fallback truncation",
            originalSize: JSON.stringify(obj).length,
            floorLocation: metadata.floorLocation || 'unknown'
        }
    }
    ];
}

function normalizelocationMetadata(meta: EndpointMetadata): EndpointMetadata {
    //Gestisci sia boolean che le stringhe
    const isFirst = meta.isFirstFloor === 'first' || meta.isFirstFloor === 'First' || meta.isFirstFloor === 'true';
    const isSecond = meta.isSecondFloor === 'second' || meta.isSecondFloor === 'Second' || meta.isSecondFloor === 'true';

    let floorLocation = meta.floorLocation;

    // Calcola floorLocation solo se non è già definito o è incoerente
    if (!floorLocation || floorLocation === 'unknown' || floorLocation === undefined) {
        if (isFirst && isSecond) floorLocation = 'both';
        else if (isFirst) floorLocation = 'first';
        else if (isSecond) floorLocation = 'second';
        else floorLocation = 'unknown';
    }

    return {
        ...meta,
        isFirstFloor: isFirst ? 'first' : 'unknown',
        isSecondFloor: isSecond ? 'second' : 'unknown',
        floorLocation: floorLocation.toLowerCase()
    };


};


