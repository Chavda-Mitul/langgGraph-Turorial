import { symbol, z } from "zod";
import { tool } from "@langchain/core/tools";

// A tool to fetch stock prices (Mocking the actual API call)
export const getStockPrice = tool(
    async ({symbol}) =>{
    console.log(`[TOOL] Fetching price for: ${symbol}`);
    // In a real app, fetch from Yahoo Finance or AlphaVantage
        const prices: Record<string,string> = {
            "AAPL": "$180.25", 
            "GOOGL": "$145.10", 
            "MSFT": "$400.00"
        }   
        return prices[symbol].toUpperCase() || "Symbol not found";
    },
    {
        name: 'get_stock_price',
        description: 'use this to get the current stock price of a compamy using ticket symbol.',
        schema: z.object({
            symbol: z.string().describe('The stock ticker symbol, e.g., AAPL')
        })
    }
);

export const tools = [getStockPrice];