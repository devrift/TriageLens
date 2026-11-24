/**
 * TriageLens AI Companion Service
 * 
 * Non-UI clinical guidance and optional audio summary generator.
 * Operates offline deterministically by default; connects to Gemini or OpenRouter
 * when API keys are configured in environment variables (.env.local).
 * 
 * Invariants:
 *   - NEVER transmits raw camera frames, video, or pixel buffers (numerical telemetry only).
 *   - NEVER logs or reveals API key secret values.
 *   - Always provides a robust, calm, deterministic offline fallback.
 */

import {
  TriageCompanionInput,
  TriageCompanionOutput,
} from '../types/triage';

declare const __GEMINI_API_KEY__: string | undefined;
declare const __OPENROUTER_API_KEY__: string | undefined;

/**
 * Safely inspect environment keys without logging or leaking secrets.
 */
function getApiKeys(): { geminiKey?: string; openRouterKey?: string } {
  let geminiKey: string | undefined;
  let openRouterKey: string | undefined;

  try {
    if (typeof __GEMINI_API_KEY__ !== 'undefined' && __GEMINI_API_KEY__) {
      geminiKey = __GEMINI_API_KEY__;
    }
    if (typeof __OPENROUTER_API_KEY__ !== 'undefined' && __OPENROUTER_API_KEY__) {
      openRouterKey = __OPENROUTER_API_KEY__;
    }
  } catch {}

  try {
    // Check Vite import.meta.env
    const metaEnv = (import.meta as any)?.env;
    if (metaEnv) {
      geminiKey = geminiKey || metaEnv.VITE_GEMINI_API_KEY || metaEnv.GEMINI_API_KEY;
      openRouterKey = openRouterKey || metaEnv.VITE_OPENROUTER_API_KEY || metaEnv.OPENROUTER_API_KEY;
    }
  } catch {}

  try {
    // Check Node process.env
    if (typeof process !== 'undefined' && process.env) {
      geminiKey = geminiKey || process.env.VITE_GEMINI_API_KEY || process.env.GEMINI_API_KEY;
      openRouterKey = openRouterKey || process.env.VITE_OPENROUTER_API_KEY || process.env.OPENROUTER_API_KEY;
    }
  } catch {}

  return { geminiKey, openRouterKey };
}

/**
 * Determine triage severity and generate calm, structured offline clinical guidance.
 */
export function generateOfflineGuidance(input: TriageCompanionInput): TriageCompanionOutput {
  const { telemetry, mode, baselineDelta } = input;
  const { heartRateBpm, respirationRateRpm, strokeAsymmetryDelta, sqiScore } = telemetry;

  const actionableAdvice: string[] = [];
  let severityLevel: TriageCompanionOutput['severityLevel'] = 'NORMAL';

  // 1. Quality / Reliability Check
  const isSqiPoor = sqiScore < 0.50;
  if (isSqiPoor) {
    severityLevel = 'INVALID';
    return {
      summaryText: 'Scan reading has high optical noise. Vitals could not be verified with high confidence.',
      guidanceText: 'Optical signal-to-noise ratio was insufficient. Please rest comfortably in a well-lit area, keep still, and repeat the scan.',
      severityLevel,
      actionableAdvice: [
        'Ensure steady lighting without rapid flickering or backlighting.',
        'Keep head or finger relaxed against the camera without moving.',
        'Repeat the 15-second scan for an accurate reading.',
      ],
      isOfflineFallback: true,
      generatedAt: Date.now(),
    };
  }

  // 2. Stroke Asymmetry (FAST Protocol) Check
  const hasSevereAsymmetry = strokeAsymmetryDelta >= 0.12 && mode === 'face';
  const hasModerateAsymmetry = strokeAsymmetryDelta >= 0.08 && strokeAsymmetryDelta < 0.12 && mode === 'face';

  // 3. Heart Rate Triage
  const isHighHr = heartRateBpm > 100;
  const isCriticalHr = heartRateBpm > 140 || heartRateBpm < 40;
  const isLowHr = heartRateBpm > 0 && heartRateBpm < 50;

  // 4. Respiration Rate Triage
  const isHighRr = respirationRateRpm > 24;
  const isLowRr = respirationRateRpm > 0 && respirationRateRpm < 10;

  if (hasSevereAsymmetry || isCriticalHr) {
    severityLevel = 'CRITICAL';
  } else if (hasModerateAsymmetry || isHighHr || isLowHr || isHighRr || isLowRr) {
    severityLevel = 'ELEVATED';
  }

  // Construct structured guidance text
  const observations: string[] = [];

  if (hasSevereAsymmetry) {
    observations.push('Observable bilateral asymmetry was detected along the facial oral commissures (FAST screen alert).');
    actionableAdvice.push('Perform the FAST check: smile, raise both arms, and speak a simple sentence.');
    actionableAdvice.push('If facial drooping or arm weakness is new or sudden, contact emergency medical services (911/112) promptly.');
  } else if (hasModerateAsymmetry) {
    observations.push('Mild facial asymmetry noted. This can be normal baseline expression or minor head angle.');
  }

  if (isCriticalHr) {
    observations.push(`Heart rate is recorded at ${heartRateBpm} BPM, which is significantly outside typical resting ranges.`);
    actionableAdvice.push('Sit down calmly, rest for 5 minutes, and seek clinical advice if experiencing chest tightness or dizziness.');
  } else if (isHighHr) {
    observations.push(`Heart rate is elevated at ${heartRateBpm} BPM (tachycardia pattern).`);
    actionableAdvice.push('Remain seated, breathe slowly, and hydrate if you have recently been active or stressed.');
  } else if (isLowHr) {
    observations.push(`Heart rate is resting at ${heartRateBpm} BPM (bradycardia pattern).`);
  } else {
    observations.push(`Heart rate is within normal resting range at ${heartRateBpm} BPM.`);
  }

  if (isHighRr) {
    observations.push(`Respiration rate is elevated at ${respirationRateRpm} breaths per minute.`);
    actionableAdvice.push('Focus on steady, deep abdominal breaths to help regulate breathing cadence.');
  } else if (isLowRr) {
    observations.push(`Respiration rate is slow at ${respirationRateRpm} breaths per minute.`);
  } else {
    observations.push(`Respiration is regular at ${respirationRateRpm} breaths per minute.`);
  }

  if (baselineDelta && baselineDelta.timeSinceBaselineMs > 0) {
    if (baselineDelta.hasElevatedHeartRate) {
      observations.push(`Heart rate has elevated by +${baselineDelta.deltaHeartRateBpm} BPM compared to your earlier baseline.`);
    }
  }

  if (actionableAdvice.length === 0) {
    actionableAdvice.push('Vitals are consistent with normal resting physiological baselines.');
    actionableAdvice.push('Save this scan to monitor trends over time.');
  }

  const summaryText = `Telemetry complete: ${heartRateBpm} BPM, ${respirationRateRpm} RPM, Asymmetry Δ ${strokeAsymmetryDelta}.`;
  const guidanceText = observations.join(' ');

  return {
    summaryText,
    guidanceText,
    severityLevel,
    actionableAdvice,
    isOfflineFallback: true,
    generatedAt: Date.now(),
  };
}

/**
 * Request online companion guidance using configured Gemini or OpenRouter endpoint,
 * gracefully falling back to deterministic offline analysis on network or credential absence.
 */
export async function getTriageGuidance(input: TriageCompanionInput): Promise<TriageCompanionOutput> {
  const { geminiKey, openRouterKey } = getApiKeys();

  // If no online connection or no API keys configured, return instant offline clinical guidance
  const isOnline = typeof navigator === 'undefined' || navigator.onLine !== false;
  if (!isOnline || (!geminiKey && !openRouterKey)) {
    return generateOfflineGuidance(input);
  }

  // Prioritize OpenRouter (gpt-4o-mini is ultra-reliable) then Gemini, cascading safely
  if (openRouterKey) {
    try {
      return await queryOpenRouterCompanion(input, openRouterKey);
    } catch {
      // If OpenRouter fails, attempt Gemini if key exists
    }
  }

  if (geminiKey) {
    try {
      return await queryGeminiCompanion(input, geminiKey);
    } catch {
      // Fall through to offline guidance
    }
  }

  return generateOfflineGuidance(input);
}

/**
 * Query Google Gemini API with numerical vital signs telemetry (zero video frames transmitted).
 */
async function queryGeminiCompanion(input: TriageCompanionInput, apiKey: string): Promise<TriageCompanionOutput> {
  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=${apiKey}`;

  const prompt = `You are TriageLens, a calm, reassuring, clinical triage AI assistant.
Analyze these non-invasive vital signs collected client-side via optical telemetry:
- Scan Mode: ${input.mode}
- Heart Rate: ${input.telemetry.heartRateBpm} BPM
- Respiration Rate: ${input.telemetry.respirationRateRpm} RPM
- Facial Asymmetry Delta: ${input.telemetry.strokeAsymmetryDelta} (normal < 0.05, alert >= 0.12)
- Signal Quality Index: ${input.telemetry.sqiScore}
${input.baselineDelta ? `- Baseline HR delta: ${input.baselineDelta.deltaHeartRateBpm} BPM` : ''}

Respond STRICTLY in valid JSON matching this schema:
{
  "summaryText": "short 1-sentence summary",
  "guidanceText": "calm, non-alarmist explanation of vitals",
  "severityLevel": "NORMAL" | "ELEVATED" | "CRITICAL" | "INVALID",
  "actionableAdvice": ["step 1", "step 2"]
}`;

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { responseMimeType: 'application/json', temperature: 0.2 },
    }),
  });

  if (!response.ok) {
    throw new Error(`Gemini request failed: ${response.status}`);
  }

  const json = await response.json();
  const rawText = json?.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!rawText) throw new Error('Empty Gemini response');

  const parsed = JSON.parse(rawText);
  return {
    summaryText: parsed.summaryText || 'Vitals telemetry analyzed.',
    guidanceText: parsed.guidanceText || 'All metrics analyzed successfully.',
    severityLevel: parsed.severityLevel || 'NORMAL',
    actionableAdvice: Array.isArray(parsed.actionableAdvice) ? parsed.actionableAdvice : ['Rest comfortably.'],
    isOfflineFallback: false,
    generatedAt: Date.now(),
  };
}

/**
 * Query OpenRouter API with numerical vital signs telemetry.
 */
async function queryOpenRouterCompanion(input: TriageCompanionInput, apiKey: string): Promise<TriageCompanionOutput> {
  const endpoint = 'https://openrouter.ai/api/v1/chat/completions';

  const systemPrompt = `You are TriageLens, a calm, non-alarmist clinical triage assistant. Provide objective, reassuring analysis in JSON with fields: summaryText, guidanceText, severityLevel ('NORMAL'|'ELEVATED'|'CRITICAL'|'INVALID'), actionableAdvice (array).`;
  const userContent = JSON.stringify({
    mode: input.mode,
    telemetry: input.telemetry,
    baselineDelta: input.baselineDelta,
  });

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: 'openai/gpt-4o-mini',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userContent },
      ],
      response_format: { type: 'json_object' },
      temperature: 0.2,
    }),
  });

  if (!response.ok) {
    throw new Error(`OpenRouter request failed: ${response.status}`);
  }

  const json = await response.json();
  const content = json?.choices?.[0]?.message?.content;
  if (!content) throw new Error('Empty OpenRouter response');

  const parsed = JSON.parse(content);
  return {
    summaryText: parsed.summaryText || 'Vitals telemetry analyzed.',
    guidanceText: parsed.guidanceText || 'Telemetry processed.',
    severityLevel: parsed.severityLevel || 'NORMAL',
    actionableAdvice: Array.isArray(parsed.actionableAdvice) ? parsed.actionableAdvice : ['Rest comfortably.'],
    isOfflineFallback: false,
    generatedAt: Date.now(),
  };
}

/**
 * Text-to-Speech (TTS) voice guidance options.
 */
export interface TtsSpeechOptions {
  rate?: number;
  pitch?: number;
  lang?: string;
  volume?: number;
}

/**
 * Speak clinical guidance audio via Web Speech API with safe offline/headless fallback.
 * Zero external network calls required.
 */
export function speakGuidance(text: string, options?: TtsSpeechOptions): Promise<boolean> {
  return new Promise((resolve) => {
    if (typeof window === 'undefined' || !window.speechSynthesis || typeof SpeechSynthesisUtterance === 'undefined') {
      // Graceful resolution in headless / Node environment
      resolve(false);
      return;
    }

    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = options?.rate ?? 0.95; // Calm, deliberate clinical pacing
      utterance.pitch = options?.pitch ?? 1.0;
      utterance.volume = options?.volume ?? 1.0;
      utterance.lang = options?.lang ?? 'en-US';

      utterance.onend = () => resolve(true);
      utterance.onerror = () => resolve(false);

      window.speechSynthesis.speak(utterance);
    } catch {
      resolve(false);
    }
  });
}

/**
 * Cancel any ongoing speech synthesis audio.
 */
export function stopSpeaking(): void {
  if (typeof window !== 'undefined' && window.speechSynthesis) {
    try {
      window.speechSynthesis.cancel();
    } catch {}
  }
}

