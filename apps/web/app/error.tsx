"use client";
export default function ErrorPage({error,reset}:{error:Error;reset:()=>void}){return <div className="error-panel"><h1>Dashboard data is unavailable</h1><p>{error.message}</p><p>Start the API, initialize the database, and import a dataset using the README commands.</p><button onClick={reset}>Try again</button></div>}
