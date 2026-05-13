# Nightly Failure Notification (Placeholder)

The nightly upstream-regression workflow runs against tt-metal `main` tip. When
the regression fails, we want a Slack notification to a designated channel.

**Status:** not implemented in the initial CI rollout. The nightly currently
runs with `continue-on-error: true` and surfaces failures only via the Actions
tab UI. Follow this checklist to add Slack alerting:

## Follow-up implementation steps

1. Create a Slack incoming-webhook URL for the destination channel.
2. Store as a repository secret named `NIGHTLY_SLACK_WEBHOOK_URL`.
3. Append a step to `.github/workflows/nightly-upstream.yml`:

   ```yaml
   - name: Notify Slack on regression
     if: failure()
     env:
       SLACK_WEBHOOK_URL: ${{ secrets.NIGHTLY_SLACK_WEBHOOK_URL }}
     run: |
       curl -X POST -H 'Content-Type: application/json' \
         --data "$(jq -n \
           --arg url "$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID" \
           --arg sha "$(cat tt-metal-pin-resolved.txt 2>/dev/null || echo unknown)" \
           '{text: ("Nightly tt-emule regression failed against tt-metal " + $sha + "\n" + $url)}')" \
         "$SLACK_WEBHOOK_URL"
   ```

4. Decide whether the PR-regression workflow should also notify on push-to-main
   failures (probably yes, to a different channel).
