export class ChatError extends Error {
  constructor(
    message: string,
    readonly requestId?: string,
  ) {
    super(message);
  }
}
