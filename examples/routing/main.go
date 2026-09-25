// Example: intent routing and confidence gating with the composable patterns.
//
//	go run ./examples/routing
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/jmwri/decide"
)

func main() {
	ctx := context.Background()
	message := "I was charged twice for my subscription this month, please reverse one payment."

	// Route: run one Choice and dispatch to the matching handler.
	intent := decide.NewChoice("What does the customer want?", decide.Options{
		{Key: "refund", Description: "Money back or reversal of a duplicate charge"},
		{Key: "cancel", Description: "Close the account or stop the subscription"},
		{Key: "help", Description: "Technical or how-to assistance"},
	})
	handlers := map[string]func(*decide.ChoiceAnswer) error{
		"refund": func(a *decide.ChoiceAnswer) error {
			fmt.Printf("-> refunds queue (confidence %.2f)\n", a.Confidence)
			return nil
		},
		"cancel": func(a *decide.ChoiceAnswer) error { fmt.Println("-> retention team"); return nil },
	}
	_, err := decide.Route(ctx, nil, message, intent, handlers, decide.RouteConfig{
		MinConfidence: 0.3,
		Default: func(a *decide.ChoiceAnswer) error {
			fmt.Printf("-> human review (%s, %.2f)\n", a.Choice, a.Confidence)
			return nil
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	// ConfidenceGate: automate what the model is sure about, escalate the rest.
	gate, err := decide.ConfidenceGate(ctx, nil, message, decide.TriagePreset(), 0.6)
	if err != nil {
		log.Fatal(err)
	}
	for id := range gate.Automatic {
		fmt.Println("automatic:", id)
	}
	for id := range gate.Escalate {
		fmt.Println("escalate: ", id)
	}
}
