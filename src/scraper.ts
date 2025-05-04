import playwright from "playwright";
import * as cheerio from "cheerio";
import { URL } from 'url';

interface ScrapingConfig {
    maxDepth: number;
    maxPages: number;
    sameDomain: boolean;
    excludePatterns: string[];
}

const defaultConfig: ScrapingConfig = {
    maxDepth: 2,
    maxPages: 20,
    sameDomain: true,
    excludePatterns: [
        '/auth/',
        '/login',
        '/logout',
        '/signin',
        '/signup',
        '/register',
        '.pdf',
        '.jpg',
        '.png',
        '.gif'
    ]
};

interface PageContent {
    url: string;
    title: string;
    content: string;
}

class WebCrawler {
    private visited = new Set<string>();
    private queue: { url: string; depth: number }[] = [];
    private browser: playwright.Browser | null = null;
    private baseUrl: string = '';
    private baseDomain: string = '';
    
    async init() {
        this.browser = await playwright.chromium.launch({ headless: true });
        console.log('🌐 Browser initialized');
    }

    private normalizeUrl(url: string): string {
        try {
            const parsed = new URL(url, this.baseUrl);
            return parsed.href.split('#')[0];  // Remove hash fragments
        } catch {
            return '';
        }
    }

    private shouldCrawl(url: string, config: ScrapingConfig): boolean {
        if (this.visited.has(url)) return false;
        
        const parsed = new URL(url);
        if (config.sameDomain && parsed.hostname !== this.baseDomain) return false;
        
        return !config.excludePatterns.some(pattern => url.includes(pattern));
    }

    private async extractLinks($: cheerio.CheerioAPI, currentUrl: string): Promise<string[]> {
        const links: string[] = [];
        $('a[href]').each((_, element) => {
            const href = $(element).attr('href');
            if (href) {
                const normalizedUrl = this.normalizeUrl(href);
                if (normalizedUrl) links.push(normalizedUrl);
            }
        });
        return [...new Set(links)];  // Remove duplicates
    }

    private async extractContent($: cheerio.CheerioAPI): Promise<{ content: string; title: string }> {
        // Get page title
        const title = $('title').text().trim() || 
                     $('h1').first().text().trim() || 
                     'Untitled Page';
        
        // Remove unwanted elements
        $('script, style, nav, footer, header, [role="navigation"], iframe, noscript').remove();
        
        // Extract main content with priority
        const selectors = [
            'main',
            'article',
            '[role="main"]',
            '.content',
            '.main-content',
            '#content',
            '.documentation',
            '.post-content'
        ];

        let content = '';
        for (const selector of selectors) {
            const element = $(selector);
            if (element.length) {
                content += element.text() + '\n';
            }
        }

        // Fallback to body if no content found
        if (!content.trim()) {
            content = $('body').text();
        }

        return {
            content: content
                .replace(/\s+/g, ' ')
                .replace(/\n\s*\n/g, '\n')
                .trim(),
            title
        };
    }

    async scrapePage(url: string, depth: number): Promise<{ content: PageContent; links: string[] }> {
        console.log(`📄 Scraping page: ${url} (depth: ${depth})`);
        const page = await this.browser!.newPage();
        
        try {
            await page.goto(url, { timeout: 30000, waitUntil: 'networkidle' });
            const html = await page.content();
            const $ = cheerio.load(html);
            
            const { content, title } = await this.extractContent($);
            const links = await this.extractLinks($, url);
            
            return { 
                content: { url, content, title },
                links 
            };
        } finally {
            await page.close();
        }
    }

    async crawl(startUrl: string, config: ScrapingConfig = defaultConfig): Promise<PageContent[]> {
        console.log('🚀 Starting crawl with config:', config);
        const contents: PageContent[] = [];
        
        try {
            await this.init();
            this.baseUrl = startUrl;
            this.baseDomain = new URL(startUrl).hostname;
            this.queue.push({ url: startUrl, depth: 0 });

            while (this.queue.length > 0 && this.visited.size < config.maxPages) {
                const { url, depth } = this.queue.shift()!;
                
                if (this.visited.has(url)) continue;
                this.visited.add(url);

                try {
                    const { content, links } = await this.scrapePage(url, depth);
                    contents.push(content);
                    console.log(`✅ Scraped ${url} (${this.visited.size}/${config.maxPages})`);

                    if (depth < config.maxDepth) {
                        // Add new links to queue
                        for (const link of links) {
                            if (this.shouldCrawl(link, config)) {
                                this.queue.push({ url: link, depth: depth + 1 });
                            }
                        }
                    }

                    // Rate limiting
                    await new Promise(r => setTimeout(r, 1000));
                } catch (error) {
                    console.error(`❌ Error scraping ${url}:`, error);
                }
            }
        } finally {
            await this.browser?.close();
        }

        return contents;
    }
}

export async function scrapeWebsite(url: string): Promise<PageContent[]> {
    const crawler = new WebCrawler();
    return await crawler.crawl(url);
}
