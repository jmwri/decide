// Example: support ticket triage with several questions in one call.
//
//	go run ./examples/triage "The export button crashes the settings page on Safari 17.2"
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/jmwri/decide"
)

func main() {
	ticket := "The export button crashes the settings page on Safari 17.2. Please fix asap."
	if len(os.Args) > 1 {
		ticket = os.Args[1]
	}

	questions := decide.Questions{
		{ID: "category", Question: decide.NewChoice("What kind of ticket is `ticket`?", decide.Options{
			{Key: "bug_report", Description: "Something is broken, degraded, or throwing errors"},
			{Key: "feature_request", Description: "Asking for something that does not exist yet"},
			{Key: "billing", Description: "Charges, invoices, payment methods, refunds"},
			{Key: "other", Description: "General inquiries or uncategorized"},
		})},
		{ID: "bug_severity", Question: decide.NewScore("How severe is the issue in `ticket`?", decide.Levels(
			"Cosmetic; no impact on core functionality",
			"Broken or degraded feature, but a workaround exists",
			"Blocking issue; no workaround exists",
		))},
		{ID: "has_repro_steps", Question: decide.NewNoul("Does `ticket` say how to reproduce the problem?", nil)},
		{ID: "refund_requested", Question: decide.NewNoul("Does the customer ask for money back or a refund?", nil)},
	}

	// All four questions are answered in one pass over the same state.
	resp, err := decide.SystemOne(context.Background(), decide.Object{{Key: "ticket", Value: ticket}}, questions)
	if err != nil {
		log.Fatal(err)
	}
	category, _ := resp.Choice("category")
	severity, _ := resp.Score("bug_severity")
	repro, _ := resp.Noul("has_repro_steps")
	refund, _ := resp.Noul("refund_requested")

	switch {
	case category.Confidence < 0.4:
		fmt.Println("route: human (unclear category)")
	case category.Choice == "bug_report" && severity.Score > 1.2 && repro.Noul > 0.5:
		fmt.Println("route: engineering (high priority)")
	case category.Choice == "bug_report":
		fmt.Println("route: bug_backlog")
	case category.Choice == "billing" && refund.Noul > 0.6:
		fmt.Println("route: refunds")
	default:
		fmt.Printf("route: support (%s)\n", category.Choice)
	}
	fmt.Printf("category=%s (%.2f) severity=%.2f repro=%.2f refund=%.2f\n",
		category.Choice, category.Confidence, severity.Score, repro.Noul, refund.Noul)
}
