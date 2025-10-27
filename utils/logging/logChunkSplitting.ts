
import { ExtendDocument } from "../../config/RAGNLP.js"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { ExtendsSeqMetadata } from "src/core/retrieval/splitters/SecondSplit2.js";



//=================================================  Logging e diagnostica  ========================================
/**It's the coordinator that:

Calls createSchemaAwareChunk for intelligent work
Calls RecursiveCharacterTextSplitter if additional splitting is needed
Should log and return results (but currently doesn't)

It's like a manager who delegates specific work to his specialists.

 */


/**    
         WHAT IT DOES: Chunk creation (preprocessing)

Purpose: Prepare documents for indexing!

When: During system setup, before use

Input: Original documents + chunking configuration

Output: Chunks optimized for the vector store


Avoid silent chunking errors that ruin retrieval

Optimize parameters based on real data instead of guesswork

Quick troubleshooting when RAG isn't performing well

Quality control ensures critical metadata isn't lost*/



//Provare o vedere RecursiveCharacterTextSplitter per splittare in chuks i documenti, è una funz di LangChain

export async function logChunkSplitting(
    documents: ExtendDocument[],
    chunkSize: number,
    chunkOverlap: number,
    recursiveDepth: number
): Promise<{

    documents: ExtendDocument[];
    deviceFamilies: Map<string, any>;
}[]> {
    console.log(`\n=== CHUNK PROCESSING STARTED ===`);
    console.log(`Processing ${documents.length} pre-chunked documents`);
    console.log(`Max chunk size: ${chunkSize}, overlap: ${chunkOverlap}`);

    const MAX_RECURSIVE_DEPTH = 5; //lIMITA LA RICORSIONE


    // ⚠️ DEBUG DUPLICAZIONE INPUT
    console.log("🔍 ANALISI INPUT logChunkSplitting:");
    const inputUUIDs = documents.map(d => d.metadata.uuid);
    const uniqueUUIDs = [...new Set(inputUUIDs)];
    console.log(`   Documenti in input: ${documents.length}`);
    console.log(`   UUID unici: ${uniqueUUIDs.length}`);


    // Trova duplicati
    const duplicateMap = new Map();
    documents.forEach((doc, idx) => {
        const key = `${doc.metadata.uuid}-${doc.metadata.chunkType}`;
        if (!duplicateMap.has(key)) duplicateMap.set(key, []);
        duplicateMap.get(key).push({ index: idx, name: doc.metadata.name });
    });

    duplicateMap.forEach((entries, key) => {
        if (entries.length > 1) {
            console.log(`🚨 DUPLICATI trovati per ${key}:`);
            entries.forEach((entry: any) => {
                console.log(`   - Index ${entry.index}: ${entry.name}`);
            });
        }
    });






    const results = [];
    const globalDeviceFamilies = new Map<string, any>();

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize,
        chunkOverlap,
        keepSeparator: true
    });


    // Statistics tracking
    let totalOriginalChunks = 0;
    let totalFinalChunks = 0;
    let chunksRequiringSplitting = 0;
    let chunksRequiringSecondSplit = 0;
    const chunkTypeStats: Record<string, number> = {};





    for (const [index, doc] of documents.entries()) {
        console.log(`\n--- Processing chunk ${index + 1}/${documents.length} ---`);
        console.log(`Chunk type: ${doc.metadata.chunkType}`);
        console.log(`Source: ${doc.metadata.source || 'unknown'}`);
        console.log(`Size: ${doc.pageContent.length} characters`);
        console.log(`Location metadata: floorLocation=${doc.metadata.floorLocation}, isFirstFloor=${doc.metadata.isFirstFloor}, isSecondFloor=${doc.metadata.isSecondFloor}`);


        // Update statistics
        totalOriginalChunks++;
        const chunkType = doc.metadata.chunkType || 'unknown';
        chunkTypeStats[chunkType] = (chunkTypeStats[chunkType] || 0) + 1;

        // Each document is already a chunk - validate and process
        let finalChunks = [doc];


        console.log("Curiosità", finalChunks.length);
        //console.log("Curiosità", JSON.stringify(finalChunks));

        // Check if this chunk needs secondary splitting
        if (doc.pageContent.length > chunkSize) {
            console.log(`⚠️ Chunk oversized (${doc.pageContent.length} > ${chunkSize}), applying secondary splitting...`);
            chunksRequiringSplitting++;

            // Prima suddivisione standard
            const subChunks = await splitter.splitDocuments([doc]);

            // Ricorsione: verifica se i sub-chunks sono ancora troppo grandi
            const recursiveChunk: ExtendDocument[] = [];


            for (const subChunk of subChunks) {
                if (subChunk.pageContent.length > chunkSize) {
                    if (recursiveDepth >= MAX_RECURSIVE_DEPTH) {
                        console.warn("MAx Recursion depth reached, skipping further splitting");
                        finalChunks = [doc];

                    } else {


                        const deeperSplit = await logChunkSplitting([{
                            ...subChunk,
                            metadata: {
                                ...subChunk.metadata,
                                isSubChunk: true,
                                parentChunkId: doc.metadata.chunkId,
                            }
                        }], chunkSize, chunkOverlap, recursiveDepth + 1);
                        //Recupera i documenti dal risultato
                        for (const result of deeperSplit) {
                            recursiveChunk.push(...result.documents);
                        }
                    }

                } else {
                    recursiveChunk.push(subChunk);
                }

            }

            finalChunks = recursiveChunk;
            //Nota: È importante che la ricorsione non si applichi all’intero set di documents,
            //  ma solo a singoli chunk che risultano ancora troppo grandi, altrimenti rischi duplicazioni o cicli infiniti.

        }


        finalChunks = finalChunks.map(chunk => {
            const criticalMetadata = {
                floorLocation: doc.metadata.floorLocation,
                isFirstFloor: doc.metadata.isFirstFloor,
                isSecondFloor: doc.metadata.isSecondFloor,
                partitionNames: doc.metadata.partitionNames,
                areaName: doc.metadata.areaName,
                areaUuid: doc.metadata.areaUuid,
                visualizationType: doc.metadata.visualizationType,
                category: doc.metadata.category,
                chunkType: doc.metadata.chunkType
            };

            return {
                ...chunk,
                metadata: {
                    ...doc.metadata, // Metadati originali come base
                    ...chunk.metadata, // Nuovi metadati
                    ...criticalMetadata // Assicura i metadati critici
                }
            };

        });

        totalFinalChunks += finalChunks.length;

        const deviceFamilies = new Map<string, any>();

        finalChunks.forEach(doc => {
            if (doc.metadata.chunkType === 'detail') {
                const deviceKey = doc.metadata.parentUuid || doc.metadata.uuid;
                const deviceName = doc.metadata.parentName || doc.metadata.name;

                // DEBUG: Verifica il sistema gerarchico
                if (doc.metadata.chunkId && doc.metadata.parentChunkId) {
                    console.log(`   • Hierarchy: ${doc.metadata.chunkId} → Parent ${doc.metadata.parentChunkId}`);
                }

                if (!deviceFamilies.has(deviceKey)) {
                    deviceFamilies.set(deviceKey, []);
                    globalDeviceFamilies.set(deviceKey, []);
                }
                deviceFamilies.get(deviceKey)!.push(doc);
                globalDeviceFamilies.get(deviceKey)!.push(doc);
            }
            // Aggiungi nel loop di processing
            console.log(`   • Parent device: ${doc.metadata.parentName || 'None'} (${doc.metadata.parentUuid || 'No UUID'})`);


        });

        // Log chunk analysis
        console.log(` Chunk ${index + 1} analysis:`);
        console.log(`   • Original size: ${doc.pageContent.length} chars`);
        console.log(`   • Chunk type: ${chunkType}`);
        console.log(`   • Final chunks: ${finalChunks.length}`);
        console.log(`   • Area info: ${doc.metadata.hasAreaInfo ? 'Yes' : 'No'}`);
        console.log(`   • Location filters: first=${doc.metadata.isFirstFloor}, second=${doc.metadata.isSecondFloor}`);
        console.log(`   • Location preserved: ${finalChunks.every(c =>
            c.metadata.floorLocation === doc.metadata.floorLocation &&
            c.metadata.isFirstFloor === doc.metadata.isFirstFloor &&
            c.metadata.isSecondFloor === doc.metadata.isSecondFloor
        ) ? '✅ Yes' : '❌ No'}`);

        results.push({

            documents: finalChunks,
            deviceFamilies: deviceFamilies
        });
        /*
        if (doc.metadata.chunkType === 'area') {
            console.log(`   • Area name: ${doc.metadata.areaNames || 'Unknown'}`);
            //console.log(`   • Devices count: ${doc.metadata.devicesCount || 0}`);
        }
 
        results.push({
            original: doc,
            chunks: finalChunks,
            deviceFamilies: deviceFamilies
        })*/
    }

    // Final statistics
    console.log(`\n=== CHUNK PROCESSING COMPLETED ===`);
    console.log(` Processing Statistics:`);
    console.log(`   • Original documents: ${documents.length}`);
    console.log(`   • Total original chunks: ${totalOriginalChunks}`);
    console.log(`   • Total final chunks: ${totalFinalChunks}`);
    console.log(`   • Chunks requiring splitting: ${chunksRequiringSplitting}`);
    console.log(`   • Chunk types distribution:`);

    Object.entries(chunkTypeStats).forEach(([type, count]) => {
        console.log(`     - ${type}: ${count}`);
    });

    console.log(`\n=== DEVICE FAMILIES SUMMARY ===`);
    console.log(`Total device families: ${globalDeviceFamilies.size}`);

    for (const [deviceKey, chunks] of globalDeviceFamilies.entries()) {
        const mainDevice = chunks.find((c: any) => !c.metadata.isSubChunk) || chunks[0];
        const parameters = chunks.filter((c: any) => c.metadata.isSubChunk);
        // const floorLocation = mainDevice.metadata.floorLocation || 'unknown';

        //console.log(`  ${mainDevice.metadata.name} (${floorLocation}): ${chunks.length} total chunks, \nall parameters: ${parameters}`);
    }



    return results;
}

function validateParentChildRelations(document: ExtendDocument[]): void {
    const parentChunks = document.filter(c => !c.metadata.isSubChunk);
    const childChunks = document.filter(c => c.metadata.isSubChunk);

    console.log(`Parent chunks: ${parentChunks.length}`);
    console.log(`Child chunks: ${childChunks.length}`);

    /*childChunks.forEach(child => {
        if (!child.metadata.parentUuid) {
            console.warn(`❌ Child chunk without parent: ${child.metadata.name}`);
        }
    });*/
    childChunks.forEach(child => {
        const parent = parentChunks.find(p => p.metadata.chunkId === child.metadata.parentChunkId);
        if (!parent) {
            console.warn(`❌ Orphaned child: ${child.metadata.name} (${child.metadata.chunkId})`);
        } else {
            console.log(`✅ Valid parent-child: ${child.metadata.name} → ${parent.metadata.name}`);
        }
    });
}

