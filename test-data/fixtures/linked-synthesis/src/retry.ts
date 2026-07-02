// [retry-budget-jitter]
// [[Retry Budget]] explains why jitter is part of the retry contract.
export function retryDelay(attempt: number): number {
  const capped = Math.min(attempt, 4);
  return 100 * capped + Math.floor(Math.random() * 25);
}
