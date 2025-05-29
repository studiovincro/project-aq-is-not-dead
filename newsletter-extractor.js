const fs = require('fs').promises;
const path = require('path');
const { generateObject, generateText } = require('ai');
const { createOpenAI } = require('@ai-sdk/openai');
const { z } = require('zod');

// GitHub Copilot models via OpenAI-compatible endpoint
const copilot = createOpenAI({
    apiKey: process.env.GITHUB_TOKEN,
    baseURL: 'https://models.inference.ai.azure.com',
    // Alternative: use gpt-4o, gpt-4o-mini, or o1-preview
});

// Zod schemas for structured extraction
const NewsletterFormatSchema = z.object({
    has_ratings: z.boolean(),
    rating_scale: z.string(),
    section_headers: z.array(z.string()),
    common_patterns: z.object({
        artist_indicator: z.string(),
        album_indicator: z.string(),
        review_start: z.string()
    })
});

const AlbumSchema = z.object({
    artist: z.string(),
    album: z.string(),
    genre: z.string().nullable(),
    label: z.string().nullable(),
    review_text: z.string().nullable(),
    rating: z.string().nullable().optional()
});

const NewsletterSchema = z.object({
    newsletter_date: z.string().nullable(),
    total_albums: z.number(),
    new_arrivals: z.array(AlbumSchema)
});

class NewsletterFormat {
    constructor({
                    hasRatings = false,
                    ratingScale = 'none',
                    sectionHeaders = [],
                    artistPatterns = [],
                    albumPatterns = [],
                    reviewPatterns = []
                } = {}) {
        this.hasRatings = hasRatings;
        this.ratingScale = ratingScale;
        this.sectionHeaders = sectionHeaders;
        this.artistPatterns = artistPatterns;
        this.albumPatterns = albumPatterns;
        this.reviewPatterns = reviewPatterns;
    }
}

class NewsletterExtractor {
    constructor(githubToken, formatCachePath = 'format_cache.json') {
        this.githubToken = githubToken;
        this.formatCachePath = formatCachePath;
        this.knownFormats = {};
        this.model = copilot('gpt-4o'); // or 'gpt-4o-mini' for faster/cheaper processing
        this.maxTokens = 5000; // Set below 8000 to allow for prompt overhead

        // Rate limiting configuration
        this.tokenUsage = [];
        this.rateLimit = 40000; // 40,000 tokens per minute
        this.rateLimitWindow = 60 * 1000; // 60 seconds in milliseconds

        this.init();
    }

    async init() {
        this.knownFormats = await this.loadFormatCache();
    }

    // Add this method to track token usage and enforce rate limits
    async enforceRateLimit(estimatedTokens) {
        const now = Date.now();

        // Remove token usage records older than the rate limit window
        this.tokenUsage = this.tokenUsage.filter(
            usage => now - usage.timestamp < this.rateLimitWindow
        );

        // Calculate total tokens used in the current window
        const tokensInWindow = this.tokenUsage.reduce(
            (sum, usage) => sum + usage.tokens, 0
        );

        // If adding these tokens would exceed the rate limit, wait
        if (tokensInWindow + estimatedTokens > this.rateLimit) {
            const oldestUsage = this.tokenUsage[0];
            const timeToWait = this.rateLimitWindow - (now - oldestUsage.timestamp) + 1000; // Add 1s buffer

            console.log(`Rate limit approaching (${tokensInWindow}/${this.rateLimit} tokens used). Waiting ${Math.round(timeToWait/1000)}s...`);

            await new Promise(resolve => setTimeout(resolve, timeToWait));

            // Recursive call to check again after waiting
            return this.enforceRateLimit(estimatedTokens);
        }

        // Record this token usage
        this.tokenUsage.push({
            tokens: estimatedTokens,
            timestamp: now
        });

        return true;
    }

    // Estimate token count (rough approximation: ~4 chars per token)
    estimateTokenCount(text) {
        return Math.ceil(text.length / 4);
    }

    // Split text into chunks of approximately maxTokens
    splitIntoChunks(text, maxTokens = this.maxTokens) {
        const chunks = [];
        let remainingText = text;

        // Find good splitting points (paragraphs, newlines)
        while (this.estimateTokenCount(remainingText) > maxTokens) {
            // Try to find paragraph breaks or other natural splitting points
            let splitIndex = -1;

            // Look for paragraph breaks within the token limit
            const approximateCharLimit = maxTokens * 4;
            const textSegment = remainingText.substring(0, Math.min(approximateCharLimit * 1.2, remainingText.length));

            // First try to split at paragraph breaks
            const paragraphMatch = textSegment.lastIndexOf('\n\n');
            if (paragraphMatch > approximateCharLimit * 0.5) {
                splitIndex = paragraphMatch + 2; // Include the newline characters
            } else {
                // Fall back to single newlines
                const newlineMatch = textSegment.lastIndexOf('\n');
                if (newlineMatch > approximateCharLimit * 0.5) {
                    splitIndex = newlineMatch + 1;
                } else {
                    // Last resort: split at a sentence
                    const sentenceMatch = textSegment.lastIndexOf('. ');
                    if (sentenceMatch > approximateCharLimit * 0.5) {
                        splitIndex = sentenceMatch + 2;
                    } else {
                        // If all else fails, hard split at token limit
                        splitIndex = approximateCharLimit;
                    }
                }
            }

            // Add chunk and continue with remaining text
            chunks.push(remainingText.substring(0, splitIndex));
            remainingText = remainingText.substring(splitIndex);
        }

        // Add the final chunk
        if (remainingText.length > 0) {
            chunks.push(remainingText);
        }

        return chunks;
    }

    async processNewsletter(filePath) {
        try {
            console.log(`Processing: ${path.basename(filePath)}`);

            // Step 1: Read file
            const text = await this.readFile(filePath);

            // Check if text needs to be chunked
            const estimatedTokens = this.estimateTokenCount(text);
            if (estimatedTokens > this.maxTokens) {
                console.log(`Newsletter is large (est. ${estimatedTokens} tokens). Splitting into chunks...`);
                return await this.processLargeNewsletter(text, filePath);
            }

            // Step 2: Format Detection & Adaptation
            const detectedFormat = await this.detectFormat(text);

            // Step 3: Structured Extraction with Vercel AI SDK
            const extractedData = await this.extractWithStructuredAI(text, detectedFormat);

            // Step 4: Validation & Correction
            const validatedResult = await this.validateAndCorrect(extractedData, text, detectedFormat);

            // Step 5: Format Learning
            await this.updateFormatKnowledge(detectedFormat, filePath);

            return {
                ...validatedResult,
                metadata: {
                    sourceFile: path.basename(filePath),
                    processedAt: new Date().toISOString(),
                    formatSignature: this.createFormatSignature(text),
                    model: 'github-copilot-gpt-4o'
                }
            };

        } catch (error) {
            console.error(`Error processing ${filePath}:`, error.message);
            return {
                error: error.message,
                sourceFile: path.basename(filePath),
                new_arrivals: []
            };
        }
    }

    async processLargeNewsletter(text, filePath) {
        console.log('Processing large newsletter in chunks...');

        // Step 1: Detect format using just the beginning of the newsletter
        const formatSample = text.substring(0, 2000); // First 2000 chars for format detection
        const detectedFormat = await this.detectFormat(formatSample);

        // Step 2: Split text into chunks
        const chunks = this.splitIntoChunks(text);
        console.log(`Split into ${chunks.length} chunks`);

        // Step 3: Process each chunk
        let allAlbums = [];
        let chunkResults = [];
        let newsletterDate = null;

        for (let i = 0; i < chunks.length; i++) {
            console.log(`Processing chunk ${i+1} of ${chunks.length}...`);

            try {
                // Create context instructions for this chunk
                const chunkPrompt = `
                This is CHUNK ${i+1} OF ${chunks.length} from a larger newsletter.
                Extract ALL album information from this section.

                ${i === 0 ? 'This is the first chunk, also extract the newsletter date if present.' : ''}
                ${i > 0 ? 'Make sure not to duplicate albums from previous chunks.' : ''}
                `;

                const chunkResult = await this.extractWithStructuredAI(
                    chunks[i],
                    detectedFormat,
                    chunkPrompt
                );

                // Store results
                chunkResults.push(chunkResult);

                // Get newsletter date from first chunk if available
                if (i === 0 && chunkResult.newsletter_date) {
                    newsletterDate = chunkResult.newsletter_date;
                }

                // Collect all albums
                if (chunkResult.new_arrivals && Array.isArray(chunkResult.new_arrivals)) {
                    allAlbums = [...allAlbums, ...chunkResult.new_arrivals];
                }

                // Avoid rate limiting
                if (i < chunks.length - 1) {
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            } catch (error) {
                console.error(`Error processing chunk ${i+1}:`, error.message);
            }
        }

        // Step 4: Combine results
        const combinedResult = {
            newsletter_date: newsletterDate,
            total_albums: allAlbums.length,
            new_arrivals: allAlbums,
            metadata: {
                sourceFile: path.basename(filePath),
                processedAt: new Date().toISOString(),
                formatSignature: this.createFormatSignature(formatSample),
                model: 'github-copilot-gpt-4o',
                processed_in_chunks: true,
                chunk_count: chunks.length
            }
        };

        // Step 5: Remove duplicates
        const uniqueAlbums = this.removeDuplicateAlbums(combinedResult.new_arrivals);
        combinedResult.new_arrivals = uniqueAlbums;
        combinedResult.total_albums = uniqueAlbums.length;

        // Step 6: Format Learning
        await this.updateFormatKnowledge(detectedFormat, filePath);

        return combinedResult;
    }

    removeDuplicateAlbums(albums) {
        const seen = new Set();
        return albums.filter(album => {
            // Create a unique key for each album
            const key = `${album.artist}|${album.album}`;

            // Check if we've seen this key before
            if (seen.has(key)) return false;

            // Add to seen set and keep this album
            seen.add(key);
            return true;
        });
    }

    async detectFormat(text) {
        // Check against known formats first
        const formatSignature = this.createFormatSignature(text);
        if (this.knownFormats[formatSignature]) {
            console.log('Using cached format');
            return new NewsletterFormat(this.knownFormats[formatSignature]);
        }

        console.log('Analyzing newsletter format...');

        try {
            // Estimate token usage for format detection
            const promptTokens = this.estimateTokenCount(text.substring(0, 1000)) + 500;

            // Enforce rate limit before making the API call
            await this.enforceRateLimit(promptTokens);

            // Use structured generation for format detection
            const { object: formatData } = await generateObject({
                model: this.model,
                schema: NewsletterFormatSchema,
                prompt: `
                Analyze this newsletter's structure and identify:
                1. Whether it contains ratings/scores for albums
                2. What rating scale is used (1-5, 1-10, A-F, stars, etc.)
                3. Main section headers present
                4. Common patterns for artist names, album titles, and reviews

                Newsletter sample (first 1000 chars):
                ${text.substring(0, 1000)}
                `,
                temperature: 0.1
            });

            return this.parseFormatResponse(formatData);
        } catch (error) {
            console.warn('Format detection failed, using fallback:', error.message);
            return new NewsletterFormat(); // Default format
        }
    }

    async extractWithStructuredAI(text, formatInfo, additionalPrompt = '') {
        try {
            console.log('Extracting structured data...');

            // Estimate token usage (prompt + completion)
            const promptTokens = this.estimateTokenCount(text) + 500; // 500 for instructions

            // Enforce rate limit before making the API call
            await this.enforceRateLimit(promptTokens);

            // Create dynamic schema based on format
            const DynamicAlbumSchema = formatInfo.hasRatings
                ? AlbumSchema.extend({ rating: z.string().nullable() })
                : AlbumSchema.omit({ rating: true });

            const DynamicNewsletterSchema = z.object({
                newsletter_date: z.string().nullable(),
                total_albums: z.number(),
                new_arrivals: z.array(DynamicAlbumSchema)
            });

            // Generate format-specific instructions
            const formatInstructions = this.generateFormatInstructions(formatInfo);

            const { object: extractedData } = await generateObject({
                model: this.model,
                schema: DynamicNewsletterSchema,
                prompt: `
                Extract all album information from this music newsletter.

                ${additionalPrompt}

                FORMAT REQUIREMENTS:
                ${formatInstructions}

                EXTRACTION RULES:
                - Extract ALL albums mentioned, not just featured ones
                - If information is missing, use null rather than guessing
                - Preserve review text exactly as written
                - Parse dates in YYYY-MM-DD format when possible
                - Count total albums accurately

                Newsletter text:
                ${text}
                `,
                temperature: 0.1
            });

            return extractedData;

        } catch (error) {
            console.error('Structured extraction failed:', error.message);

            // Fallback to text generation if structured fails
            return await this.extractWithTextGeneration(text, formatInfo, additionalPrompt);
        }
    }

    async extractWithTextGeneration(text, formatInfo, additionalPrompt = '') {
        console.log('Using fallback text generation...');

        try {
            // Estimate token usage for text generation
            const promptTokens = this.estimateTokenCount(text) + 600; // 600 for instructions

            // Enforce rate limit before making the API call
            await this.enforceRateLimit(promptTokens);

            const formatInstructions = this.generateFormatInstructions(formatInfo);

            const { text: result } = await generateText({
                model: this.model,
                prompt: `
                Extract structured data from this music newsletter and return ONLY valid JSON.

                ${additionalPrompt}

                FORMAT REQUIREMENTS:
                ${formatInstructions}

                Required JSON structure:
                {
                  "newsletter_date": "YYYY-MM-DD or null",
                  "total_albums": number,
                  "new_arrivals": [
                    {
                      "artist": "string",
                      "album": "string",
                      "genre": "string or null",
                      "label": "string or null",
                      "review_text": "string or null"${formatInfo.hasRatings ? ',\n      "rating": "string or null"' : ''}
                    }
                  ]
                }

                Newsletter text:
                ${text}
                `,
                temperature: 0.1
            });

            return JSON.parse(this.cleanJsonResponse(result));

        } catch (error) {
            console.error('Text generation fallback failed:', error.message);
            return {
                error: error.message,
                newsletter_date: null,
                total_albums: 0,
                new_arrivals: []
            };
        }
    }

    generateFormatInstructions(formatInfo) {
        const instructions = [];

        if (formatInfo.hasRatings) {
            instructions.push(`- Look for ratings on ${formatInfo.ratingScale} scale`);
        }

        if (formatInfo.sectionHeaders.length > 0) {
            instructions.push(`- Main sections: ${formatInfo.sectionHeaders.join(', ')}`);
        }

        instructions.push('- Extract ALL albums mentioned, not just featured ones');
        instructions.push('- If information is missing or unclear, use null rather than guessing');
        instructions.push('- Preserve original review text exactly as written');

        return instructions.join('\n');
    }

    async validateAndCorrect(result, originalText, formatInfo) {
        if (result.error) {
            return result;
        }

        // Basic validation checks
        if (!result.new_arrivals || !Array.isArray(result.new_arrivals)) {
            console.warn('Invalid new_arrivals format, attempting correction...');

            try {
                // Estimate token usage for correction
                const promptTokens = this.estimateTokenCount(originalText.substring(0, 1500)) + 700;

                // Enforce rate limit before making the API call
                await this.enforceRateLimit(promptTokens);

                const { object: corrected } = await generateObject({
                    model: this.model,
                    schema: NewsletterSchema,
                    prompt: `
                    Fix this malformed newsletter extraction. The new_arrivals should be a proper array.

                    Problematic extraction:
                    ${JSON.stringify(result, null, 2)}

                    Original text sample:
                    ${originalText.substring(0, 1500)}

                    Return corrected data with proper structure.
                    `,
                    temperature: 0.1
                });

                return corrected;
            } catch (correctionError) {
                console.error('Correction failed:', correctionError.message);
                return {
                    ...result,
                    validation_error: 'Could not correct extraction format'
                };
            }
        }

        // Validate album count
        if (result.new_arrivals.length === 0) {
            console.warn('No albums extracted - this might indicate a parsing issue');
        }

        // Set total_albums if not present or incorrect
        if (!result.total_albums || result.total_albums !== result.new_arrivals.length) {
            result.total_albums = result.new_arrivals.length;
        }

        return result;
    }

    async updateFormatKnowledge(formatInfo, filePath) {
        const formatSignature = this.createFormatSignature(await this.readFile(filePath));
        this.knownFormats[formatSignature] = {
            hasRatings: formatInfo.hasRatings,
            ratingScale: formatInfo.ratingScale,
            sectionHeaders: formatInfo.sectionHeaders,
            artistPatterns: formatInfo.artistPatterns,
            albumPatterns: formatInfo.albumPatterns,
            reviewPatterns: formatInfo.reviewPatterns,
            lastUsed: new Date().toISOString(),
            sourceFile: path.basename(filePath)
        };

        await this.saveFormatCache();
    }

    createFormatSignature(text) {
        // Enhanced signature for better format detection
        const hasArtistColon = text.includes('Artist:');
        const hasAlbumColon = text.includes('Album:');
        const hasStars = text.includes('★') || text.includes('*');
        const hasNumbers = /\b[1-9]\/[1-9]\b/.test(text);
        const hasNewArrivals = /new arrivals/i.test(text);
        const hasReviews = /review[s]?:/i.test(text);
        const hasGenre = /genre:/i.test(text);
        const hasFormat = /\b(cd|lp|vinyl|cassette|book)\b/i.test(text);
        const hasPrice = /\$\d+|\d+\.\d+/.test(text);

        return `${hasArtistColon}_${hasAlbumColon}_${hasStars}_${hasNumbers}_${hasNewArrivals}_${hasReviews}_${hasGenre}_${hasFormat}_${hasPrice}`;
    }

    parseFormatResponse(formatData) {
        const commonPatterns = formatData.common_patterns || {};

        return new NewsletterFormat({
            hasRatings: formatData.has_ratings || false,
            ratingScale: formatData.rating_scale || 'none',
            sectionHeaders: formatData.section_headers || [],
            artistPatterns: [commonPatterns.artist_indicator || ''].filter(Boolean),
            albumPatterns: [commonPatterns.album_indicator || ''].filter(Boolean),
            reviewPatterns: [commonPatterns.review_start || ''].filter(Boolean)
        });
    }

    cleanJsonResponse(text) {
        // Remove markdown code blocks if present
        text = text.replace(/```json\s*|\s*```/g, '');

        // Remove any leading/trailing whitespace
        text = text.trim();

        // Find the first { and last } to extract just the JSON
        const firstBrace = text.indexOf('{');
        const lastBrace = text.lastIndexOf('}');

        if (firstBrace !== -1 && lastBrace !== -1) {
            text = text.substring(firstBrace, lastBrace + 1);
        }

        return text;
    }

    async loadFormatCache() {
        try {
            const data = await fs.readFile(this.formatCachePath, 'utf8');
            return JSON.parse(data);
        } catch (error) {
            console.log('No format cache found, starting fresh');
            return {};
        }
    }

    async saveFormatCache() {
        try {
            await fs.writeFile(
                this.formatCachePath,
                JSON.stringify(this.knownFormats, null, 2)
            );
        } catch (error) {
            console.error('Failed to save format cache:', error.message);
        }
    }

    async readFile(filePath) {
        return await fs.readFile(filePath, 'utf8');
    }

    // Enhanced batch processing with progress tracking and rate limit handling
    async processNewsletterDirectory(directoryPath, outputDir = 'output') {
        try {
            await fs.mkdir(outputDir, { recursive: true });

            const files = await fs.readdir(directoryPath);
            const textFiles = files.filter(file =>
                file.endsWith('.txt') || file.endsWith('.md')
            );

            console.log(`Found ${textFiles.length} files to process`);

            const results = [];
            let processed = 0;

            for (const file of textFiles) {
                const filePath = path.join(directoryPath, file);

                console.log(`\n[${processed + 1}/${textFiles.length}] Processing ${file}...`);

                try {
                    const result = await this.processNewsletter(filePath);

                    // Save individual result
                    const outputFile = path.join(outputDir, `${path.parse(file).name}_extracted.json`);
                    await fs.writeFile(outputFile, JSON.stringify(result, null, 2));

                    results.push({
                        file: file,
                        success: !result.error,
                        albumCount: result.new_arrivals ? result.new_arrivals.length : 0,
                        data: result
                    });
                } catch (error) {
                    console.error(`Failed to process ${file}:`, error.message);
                    results.push({
                        file: file,
                        success: false,
                        albumCount: 0,
                        error: error.message
                    });
                }

                processed++;

                // Adaptive rate limiting - more wait time as we process more files
                if (processed < textFiles.length) {
                    const waitTime = 3000 + (Math.floor(processed / 5) * 1000); // Increase wait time every 5 files
                    console.log(`Waiting ${waitTime/1000}s before next file...`);
                    await new Promise(resolve => setTimeout(resolve, waitTime));
                }
            }

            // Save combined results
            await fs.writeFile(
                path.join(outputDir, 'all_newsletters_extracted.json'),
                JSON.stringify(results, null, 2)
            );

            // Generate summary
            const summary = this.generateProcessingSummary(results);
            await fs.writeFile(
                path.join(outputDir, 'processing_summary.json'),
                JSON.stringify(summary, null, 2)
            );

            console.log('\n🎉 Processing complete!');
            console.log(`📁 Processed: ${results.length} files`);
            console.log(`✅ Successful: ${results.filter(r => r.success).length}`);
            console.log(`💿 Total albums extracted: ${results.reduce((sum, r) => sum + r.albumCount, 0)}`);
            console.log(`📊 Results saved to: ${outputDir}/`);

            return results;

        } catch (error) {
            console.error('Batch processing failed:', error.message);
            throw error;
        }
    }

    generateProcessingSummary(results) {
        const successful = results.filter(r => r.success);
        const failed = results.filter(r => !r.success);

        return {
            totalFiles: results.length,
            successful: successful.length,
            failed: failed.length,
            totalAlbumsExtracted: results.reduce((sum, r) => sum + r.albumCount, 0),
            averageAlbumsPerNewsletter: successful.length > 0
                ? Math.round(successful.reduce((sum, r) => sum + r.albumCount, 0) / successful.length)
                : 0,
            formatsCached: Object.keys(this.knownFormats).length,
            failedFiles: failed.map(f => ({ file: f.file, error: f.data?.error || f.error })),
            processedAt: new Date().toISOString(),
            modelUsed: 'github-copilot-gpt-4o'
        };
    }
}

// Usage Example
async function main() {
    if (!process.env.GITHUB_TOKEN) {
        console.error('Please set GITHUB_TOKEN environment variable');
        process.exit(1);
    }

    const extractor = new NewsletterExtractor(process.env.GITHUB_TOKEN);

    // Process entire directory
    try {
        await extractor.processNewsletterDirectory('newsletters/', 'extracted_output');
    } catch (error) {
        console.error('Processing failed:', error.message);
    }
}

// Export for use as module
module.exports = {
    NewsletterExtractor,
    NewsletterFormat
};

// Run if called directly
if (require.main === module) {
    main().catch(console.error);
}