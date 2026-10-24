import { timingSafeEqual } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";

function matches(actual: string, expected: string): boolean {
  const left = Buffer.from(actual);
  const right = Buffer.from(expected);
  return left.length === right.length && timingSafeEqual(left, right);
}

export function proxy(request: NextRequest) {
  const username = process.env.DASHBOARD_USERNAME;
  const password = process.env.DASHBOARD_PASSWORD;
  if (!username || !password) {
    return new NextResponse("Dashboard credentials are not configured", { status: 503 });
  }
  const authorization = request.headers.get("authorization") ?? "";
  if (authorization.startsWith("Basic ")) {
    try {
      const decoded = Buffer.from(authorization.slice(6), "base64").toString("utf8");
      const separator = decoded.indexOf(":");
      if (separator >= 0 && matches(decoded.slice(0, separator), username) &&
          matches(decoded.slice(separator + 1), password)) {
        return NextResponse.next();
      }
    } catch { /* malformed credentials receive the same challenge */ }
  }
  return new NextResponse("Authentication required", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="Customer Churn Intelligence"' },
  });
}

export const config = { matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"] };
