type Chunk = {
    text?: string,
    page_content?: string,
    metadata?: Record<string, any>;
};

function debugDeviceChunk(label: string, chunk: Chunk) {
    console.log(`\n=== DEBUG [${label}] ===`);
    console.log("Name:", chunk.metadata?.name);
    console.log("Text:", chunk.text || chunk.page_content);
    console.log("Metadata.category:", chunk.metadata?.category);
    //console.log("Metadata.uuid:", chunk.metadata?.uuid);
    //console.log("Metadata.chunkType:", chunk.metadata?.chunkType);
    // console.log("Metadata.type:", chunk.metadata?.type);
    console.log("============================\n");
}

export function debugDeviceType(label: string, category: any, resolvedType: string) {
    console.log(`\n>>> DEBUG [${label}] getDeviceType`);
    console.log("Input category:", category, `(${typeof category})`);
    console.log("Resolved type:", resolvedType);
    console.log("<<< END DEBUG\n");
}
