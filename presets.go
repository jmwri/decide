package decide

// Pre-configured decision suites for high-frequency operational pipelines.

// TriagePreset is customer support ticket triage and routing.
func TriagePreset() Questions {
	return Questions{
		{"intent", NewChoice("What is the primary customer intent in the message?", Options{
			{"refund", "Requesting money back, refund, or duplicate billing reversal"},
			{"technical_help", "Reporting a bug, API error, 500 downtime, or integration issue"},
			{"billing_question", "Questions about invoices, subscription plans, or payment methods"},
			{"cancellation", "Requesting account closure, cancellation, or downgrading"},
			{"general_info", "Inquiring about documentation, pricing tiers, or how-to guidance"},
		})},
		{"is_urgent", NewNoul("Does the customer communicate extreme urgency, critical outage, or impending deadline?", &NoulCriteria{
			True:  "Urgent, production down, emergency, immediate attention needed",
			False: "Routine question, low priority, general feedback",
		})},
		{"frustration", NewScore("Rate the customer frustration level.", Levels(
			"Calm and polite",
			"Slightly concerned or asking for status",
			"Visibly frustrated or annoyed",
			"Extremely angry, threatening legal action or cancellation",
		))},
		{"churn_risk", NewNoul("Does the message indicate high risk of the customer leaving or churning?", &NoulCriteria{
			True:  "Threatening to switch to competitors, cancel contract, or stop using product",
			False: "Committed user asking for help, no mention of leaving",
		})},
	}
}

// EmailPreset is inbound email triage and threat filtering. Pass nil
// customCategories for the default destination teams.
func EmailPreset(customCategories Options) Questions {
	categories := customCategories
	if len(categories) == 0 {
		categories = Options{
			{"billing", "Invoices, payments, credit cards, pricing questions"},
			{"engineering", "Bug reports, API failures, stack traces, system outages"},
			{"sales", "Enterprise demos, contract inquiries, volume discounts"},
			{"security", "Phishing reports, suspicious access, vulnerability disclosures"},
			{"general", "General questions or uncategorized inquiries"},
		}
	}
	return Questions{
		{"destination", NewChoice("Which internal team should handle this email?", categories)},
		{"is_spam_or_phishing", NewNoul("Is this email an unsolicited sales pitch, scam, or phishing attempt?", &NoulCriteria{
			True:  "Spam, promotional blast, credential harvesting, phishing",
			False: "Legitimate user or customer inquiry",
		})},
		{"priority", NewScore("What priority level should be assigned to this email?", Levels(
			"Low: Newsletter, informational, no action required",
			"Medium: Standard inquiry with 24-48hr SLA",
			"High: Blocking issue affecting paying customer",
			"Critical: Security breach, legal threat, or severe production impact",
		))},
	}
}

// ModerationPreset is user-generated content and trust & safety moderation.
func ModerationPreset() Questions {
	return Questions{
		{"policy_violation", NewChoice("Does this content violate acceptable use policies?", Options{
			{"clean", "Content is safe, constructive, and follows community guidelines"},
			{"harassment", "Direct personal attacks, bullying, threats, or hate speech"},
			{"spam", "Repetitive links, commercial spam, crypto scams, or bot text"},
			{"sensitive", "Explicit adult content, graphic violence, or illegal goods"},
		})},
		{"should_block", NewNoul("Should this content be immediately blocked from publication?", &NoulCriteria{
			True:  "Clear violation requiring immediate rejection",
			False: "Safe or borderline content that can be published or reviewed",
		})},
		{"severity", NewScore("Rate the severity of the content risk.", Levels(
			"Safe: Compliant content",
			"Low: Minor profanity or mild uncivil behavior",
			"Medium: Aggressive tone, self-promotion, or borderline spam",
			"High: Severe violation, harassment, or malicious payload",
		))},
	}
}

// SecurityPreset is security event triage and anomaly assessment.
func SecurityPreset() Questions {
	return Questions{
		{"event_type", NewChoice("Classify the observed security or authentication anomaly.", Options{
			{"benign", "Expected user activity, legitimate IP change, or normal login"},
			{"credential_stuffing", "Rapid succession of failed logins across multiple accounts"},
			{"brute_force", "Repeated failed attempts targeting a single high-value account"},
			{"privilege_escalation", "Attempting unauthorized administrative or sudo operations"},
			{"data_exfiltration", "Abnormal volume of export requests or bulk database downloads"},
		})},
		{"is_threat", NewNoul("Does this state represent an active, confirmed malicious security threat?", &NoulCriteria{
			True:  "Active cyber attack, intrusion, or unauthorized compromise",
			False: "Normal operational glitch, user error, or benign variance",
		})},
		{"severity", NewScore("Rate the incident severity.", Levels(
			"Informational: Logged for audit trail, no action",
			"Warning: Suspicious variance, rate-limit triggered",
			"Elevated: Incident responder paged for triage",
			"Critical: Active breach, immediate token revocation and IP ban",
		))},
	}
}
