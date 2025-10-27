"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
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
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.logChunkSplitting = logChunkSplitting;
var text_splitter_1 = require("langchain/text_splitter");
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
function logChunkSplitting(documents, chunkSize, chunkOverlap, recursiveDepth) {
    return __awaiter(this, void 0, void 0, function () {
        var MAX_RECURSIVE_DEPTH, inputUUIDs, uniqueUUIDs, duplicateMap, results, globalDeviceFamilies, splitter, totalOriginalChunks, totalFinalChunks, chunksRequiringSplitting, chunksRequiringSecondSplit, chunkTypeStats, _loop_1, _i, _a, _b, index, doc, _c, _d, _e, deviceKey, chunks, mainDevice, parameters;
        return __generator(this, function (_f) {
            switch (_f.label) {
                case 0:
                    console.log("\n=== CHUNK PROCESSING STARTED ===");
                    console.log("Processing ".concat(documents.length, " pre-chunked documents"));
                    console.log("Max chunk size: ".concat(chunkSize, ", overlap: ").concat(chunkOverlap));
                    MAX_RECURSIVE_DEPTH = 5;
                    // ⚠️ DEBUG DUPLICAZIONE INPUT
                    console.log("🔍 ANALISI INPUT logChunkSplitting:");
                    inputUUIDs = documents.map(function (d) { return d.metadata.uuid; });
                    uniqueUUIDs = __spreadArray([], new Set(inputUUIDs), true);
                    console.log("   Documenti in input: ".concat(documents.length));
                    console.log("   UUID unici: ".concat(uniqueUUIDs.length));
                    duplicateMap = new Map();
                    documents.forEach(function (doc, idx) {
                        var key = "".concat(doc.metadata.uuid, "-").concat(doc.metadata.chunkType);
                        if (!duplicateMap.has(key))
                            duplicateMap.set(key, []);
                        duplicateMap.get(key).push({ index: idx, name: doc.metadata.name });
                    });
                    duplicateMap.forEach(function (entries, key) {
                        if (entries.length > 1) {
                            console.log("\uD83D\uDEA8 DUPLICATI trovati per ".concat(key, ":"));
                            entries.forEach(function (entry) {
                                console.log("   - Index ".concat(entry.index, ": ").concat(entry.name));
                            });
                        }
                    });
                    results = [];
                    globalDeviceFamilies = new Map();
                    splitter = new text_splitter_1.RecursiveCharacterTextSplitter({
                        chunkSize: chunkSize,
                        chunkOverlap: chunkOverlap,
                        keepSeparator: true
                    });
                    totalOriginalChunks = 0;
                    totalFinalChunks = 0;
                    chunksRequiringSplitting = 0;
                    chunksRequiringSecondSplit = 0;
                    chunkTypeStats = {};
                    _loop_1 = function (index, doc) {
                        var chunkType, finalChunks, subChunks, recursiveChunk, _g, subChunks_1, subChunk, deeperSplit, _h, deeperSplit_1, result, deviceFamilies;
                        return __generator(this, function (_j) {
                            switch (_j.label) {
                                case 0:
                                    console.log("\n--- Processing chunk ".concat(index + 1, "/").concat(documents.length, " ---"));
                                    console.log("Chunk type: ".concat(doc.metadata.chunkType));
                                    console.log("Source: ".concat(doc.metadata.source || 'unknown'));
                                    console.log("Size: ".concat(doc.pageContent.length, " characters"));
                                    console.log("Location metadata: floorLocation=".concat(doc.metadata.floorLocation, ", isFirstFloor=").concat(doc.metadata.isFirstFloor, ", isSecondFloor=").concat(doc.metadata.isSecondFloor));
                                    // Update statistics
                                    totalOriginalChunks++;
                                    chunkType = doc.metadata.chunkType || 'unknown';
                                    chunkTypeStats[chunkType] = (chunkTypeStats[chunkType] || 0) + 1;
                                    finalChunks = [doc];
                                    console.log("Curiosità", finalChunks.length);
                                    if (!(doc.pageContent.length > chunkSize)) return [3 /*break*/, 9];
                                    console.log("\u26A0\uFE0F Chunk oversized (".concat(doc.pageContent.length, " > ").concat(chunkSize, "), applying secondary splitting..."));
                                    chunksRequiringSplitting++;
                                    return [4 /*yield*/, splitter.splitDocuments([doc])];
                                case 1:
                                    subChunks = _j.sent();
                                    recursiveChunk = [];
                                    _g = 0, subChunks_1 = subChunks;
                                    _j.label = 2;
                                case 2:
                                    if (!(_g < subChunks_1.length)) return [3 /*break*/, 8];
                                    subChunk = subChunks_1[_g];
                                    if (!(subChunk.pageContent.length > chunkSize)) return [3 /*break*/, 6];
                                    if (!(recursiveDepth >= MAX_RECURSIVE_DEPTH)) return [3 /*break*/, 3];
                                    console.warn("MAx Recursion depth reached, skipping further splitting");
                                    finalChunks = [doc];
                                    return [3 /*break*/, 5];
                                case 3: return [4 /*yield*/, logChunkSplitting([__assign(__assign({}, subChunk), { metadata: __assign(__assign({}, subChunk.metadata), { isSubChunk: true, parentChunkId: doc.metadata.chunkId }) })], chunkSize, chunkOverlap, recursiveDepth + 1)];
                                case 4:
                                    deeperSplit = _j.sent();
                                    //Recupera i documenti dal risultato
                                    for (_h = 0, deeperSplit_1 = deeperSplit; _h < deeperSplit_1.length; _h++) {
                                        result = deeperSplit_1[_h];
                                        recursiveChunk.push.apply(recursiveChunk, result.documents);
                                    }
                                    _j.label = 5;
                                case 5: return [3 /*break*/, 7];
                                case 6:
                                    recursiveChunk.push(subChunk);
                                    _j.label = 7;
                                case 7:
                                    _g++;
                                    return [3 /*break*/, 2];
                                case 8:
                                    finalChunks = recursiveChunk;
                                    _j.label = 9;
                                case 9:
                                    finalChunks = finalChunks.map(function (chunk) {
                                        var criticalMetadata = {
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
                                        return __assign(__assign({}, chunk), { metadata: __assign(__assign(__assign({}, doc.metadata), chunk.metadata), criticalMetadata // Assicura i metadati critici
                                            ) });
                                    });
                                    totalFinalChunks += finalChunks.length;
                                    deviceFamilies = new Map();
                                    finalChunks.forEach(function (doc) {
                                        if (doc.metadata.chunkType === 'detail') {
                                            var deviceKey = doc.metadata.parentUuid || doc.metadata.uuid;
                                            var deviceName = doc.metadata.parentName || doc.metadata.name;
                                            // DEBUG: Verifica il sistema gerarchico
                                            if (doc.metadata.chunkId && doc.metadata.parentChunkId) {
                                                console.log("   \u2022 Hierarchy: ".concat(doc.metadata.chunkId, " \u2192 Parent ").concat(doc.metadata.parentChunkId));
                                            }
                                            if (!deviceFamilies.has(deviceKey)) {
                                                deviceFamilies.set(deviceKey, []);
                                                globalDeviceFamilies.set(deviceKey, []);
                                            }
                                            deviceFamilies.get(deviceKey).push(doc);
                                            globalDeviceFamilies.get(deviceKey).push(doc);
                                        }
                                        // Aggiungi nel loop di processing
                                        console.log("   \u2022 Parent device: ".concat(doc.metadata.parentName || 'None', " (").concat(doc.metadata.parentUuid || 'No UUID', ")"));
                                    });
                                    // Log chunk analysis
                                    console.log(" Chunk ".concat(index + 1, " analysis:"));
                                    console.log("   \u2022 Original size: ".concat(doc.pageContent.length, " chars"));
                                    console.log("   \u2022 Chunk type: ".concat(chunkType));
                                    console.log("   \u2022 Final chunks: ".concat(finalChunks.length));
                                    console.log("   \u2022 Area info: ".concat(doc.metadata.hasAreaInfo ? 'Yes' : 'No'));
                                    console.log("   \u2022 Location filters: first=".concat(doc.metadata.isFirstFloor, ", second=").concat(doc.metadata.isSecondFloor));
                                    console.log("   \u2022 Location preserved: ".concat(finalChunks.every(function (c) {
                                        return c.metadata.floorLocation === doc.metadata.floorLocation &&
                                            c.metadata.isFirstFloor === doc.metadata.isFirstFloor &&
                                            c.metadata.isSecondFloor === doc.metadata.isSecondFloor;
                                    }) ? '✅ Yes' : '❌ No'));
                                    results.push({
                                        documents: finalChunks,
                                        deviceFamilies: deviceFamilies
                                    });
                                    return [2 /*return*/];
                            }
                        });
                    };
                    _i = 0, _a = documents.entries();
                    _f.label = 1;
                case 1:
                    if (!(_i < _a.length)) return [3 /*break*/, 4];
                    _b = _a[_i], index = _b[0], doc = _b[1];
                    return [5 /*yield**/, _loop_1(index, doc)];
                case 2:
                    _f.sent();
                    _f.label = 3;
                case 3:
                    _i++;
                    return [3 /*break*/, 1];
                case 4:
                    // Final statistics
                    console.log("\n=== CHUNK PROCESSING COMPLETED ===");
                    console.log(" Processing Statistics:");
                    console.log("   \u2022 Original documents: ".concat(documents.length));
                    console.log("   \u2022 Total original chunks: ".concat(totalOriginalChunks));
                    console.log("   \u2022 Total final chunks: ".concat(totalFinalChunks));
                    console.log("   \u2022 Chunks requiring splitting: ".concat(chunksRequiringSplitting));
                    console.log("   \u2022 Chunk types distribution:");
                    Object.entries(chunkTypeStats).forEach(function (_a) {
                        var type = _a[0], count = _a[1];
                        console.log("     - ".concat(type, ": ").concat(count));
                    });
                    console.log("\n=== DEVICE FAMILIES SUMMARY ===");
                    console.log("Total device families: ".concat(globalDeviceFamilies.size));
                    for (_c = 0, _d = globalDeviceFamilies.entries(); _c < _d.length; _c++) {
                        _e = _d[_c], deviceKey = _e[0], chunks = _e[1];
                        mainDevice = chunks.find(function (c) { return !c.metadata.isSubChunk; }) || chunks[0];
                        parameters = chunks.filter(function (c) { return c.metadata.isSubChunk; });
                        // const floorLocation = mainDevice.metadata.floorLocation || 'unknown';
                        //console.log(`  ${mainDevice.metadata.name} (${floorLocation}): ${chunks.length} total chunks, \nall parameters: ${parameters}`);
                    }
                    return [2 /*return*/, results];
            }
        });
    });
}
function validateParentChildRelations(document) {
    var parentChunks = document.filter(function (c) { return !c.metadata.isSubChunk; });
    var childChunks = document.filter(function (c) { return c.metadata.isSubChunk; });
    console.log("Parent chunks: ".concat(parentChunks.length));
    console.log("Child chunks: ".concat(childChunks.length));
    /*childChunks.forEach(child => {
        if (!child.metadata.parentUuid) {
            console.warn(`❌ Child chunk without parent: ${child.metadata.name}`);
        }
    });*/
    childChunks.forEach(function (child) {
        var parent = parentChunks.find(function (p) { return p.metadata.chunkId === child.metadata.parentChunkId; });
        if (!parent) {
            console.warn("\u274C Orphaned child: ".concat(child.metadata.name, " (").concat(child.metadata.chunkId, ")"));
        }
        else {
            console.log("\u2705 Valid parent-child: ".concat(child.metadata.name, " \u2192 ").concat(parent.metadata.name));
        }
    });
}
