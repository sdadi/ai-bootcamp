# Checkout Error Codes Runbook (Support and Engineering)

Use this runbook when a customer reports an error code at checkout or in the payments flow. Codes are shown to the customer in the app as "Something went wrong (code XXXX)".

## E101 - Card declined by issuing bank

The customer's bank refused the authorisation. Nimbu Mart cannot override this. Ask the customer to retry with another card or pay by UPI. No money is debited.

## E102 - Payment gateway timeout

The payment gateway did not respond within 30 seconds. The order is kept in "Payment Pending" for 15 minutes while the reconciliation job (runs every 5 minutes) checks the final status with the bank.

- If the bank confirms the payment, the order is confirmed automatically.
- If money was debited but the order is not confirmed after 15 minutes, the amount is auto-refunded to the source within 48 hours.
- Do not ask the customer to pay again before the 15 minutes are over, or they may be charged twice.

## E103 - UPI collect request expired

The customer did not approve the UPI collect request within 5 minutes. No money is debited. Ask the customer to place the order again and approve the request in their UPI app.

## E104 - Inventory reservation failed

The item went out of stock between adding to cart and paying. If any money was debited, it is auto-refunded within 24 hours.

## E105 - Pincode not serviceable

We do not deliver to this pincode for this item. Large appliances and Nimbu Express have smaller serviceable areas than standard delivery.

## E107 - Coupon invalid or expired

The coupon code has expired, has reached its usage limit, or does not apply to the items in the cart. Coupons cannot be applied to gift cards.

## E110 - Wallet balance mismatch

The Nimbu Wallet balance shown to the customer does not match the ledger. Do not adjust the balance manually. Escalate to the Payments on-call engineer in the #payments-oncall channel as a P1 incident. P1 incidents must be acknowledged within 15 minutes.

## ERR_UPI_PENDING - UPI status unknown

The UPI app returned "pending". Treat it like E102: wait for reconciliation, do not retry payment for 15 minutes.
