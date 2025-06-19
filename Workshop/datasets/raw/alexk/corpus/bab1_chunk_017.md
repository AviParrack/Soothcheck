# 8/5/24

- As humans we want disconfirming evidence before making a choice, but not after making it.

  - When choosing between options we’re open to disconfirming evidence, and even seek it out.

  - But once we’ve decided and there’s no going back and we’ve bound ourselves to the option we chose we become closed to disconfirming feedback.

  - The more that it feels like there's no going back ("we all agreed as a 1000 person organization that this was the right approach, I can't re-litigate it") the more we don't want disconfirming evidence.

    - A kind of kayfabe.

    - It's easier to be wrong but consistent than to be right but inconsistent.

  - This shift before and after making the choice is profound but subtle.

    - It’s easy to miss that we even change our priorities at all.

  - But disconfirming evidence is always a good idea.

    - If we don’t get disconfirming evidence, it’s possible the idea that we’ve committed to will be very expensive for very little payoff… or could even lead to a game over situation.

    - The only time that disconfirming evidence isn’t important is if there are many “good enough” solutions and it’s more important to pick *one* than to pick the best one.

    - We often feel that we can’t go back and open up a choice we made… but it’s almost always possible to go back.

    - And we can *always* go back. It's never too late for disconfirming evidence.

- Imagine a system not focused around apps that do things with your data, but data that can do things.

  - It's your data, but intelligent.

  - Able to work on your behalf.

  - It can brainstorm with you, morph itself based on a conversation with you.

- A great way to triangulate on software you want: compare to existing apps.

  - "I want to make an app for cofounders to find each other."

  - "OK, like tinder for cofounders"

  - "No, less like Tinder because the way they look doesn't matter so it shouldn’t be as photos centric".

  - “So more like LinkedIn for cofounders?”

  - “I guess, but less about strutting and resume building, more about the kinds of personality overlaps.”

  - When you can take known existing exemplars that are very different, you get very fast triangulation by comparing to them.

- Imagine being able to do animal husbandry with apps.

  - “Take the UX of Tinder and combine it with the data in my vacation wishlist”

- A mini-app I want: something to help apply basic best practices to my personal finance.

  - Today this is very hard, because no single financial actor has a complete picture of your financial situation.

    - To do that, you’d need a lot of connectors… and you’d also have to *really* trust the place that holds your data to not leak it or sell it or muck with it.

  - Creating a place to have all of your financial data and apply best practices to it would be extremely expensive to build in today’s laws of physics.

    - The only companies willing to build it, in this current laws of physics, are companies who think they can extract more from you as a customer if they do.

    - For example, entities with a vested interest to push you towards a high-margin offering of theirs.

  - A paper by Thomas Philippon called “The Great Reversal: How America Gave Up on Free Markets" asks “why is there no Walmart of Finance?”

  - Maybe if the laws of physics changed for building software it could be possible.

- There’s a gap between Anthropic’s Artifacts and OpenAI’s GPTs.

  - Anthropic Artifacts makes it super simple to create a little sandboxed live demo app with whatever UX you want that you can share with anyone you want.

    - But the Artifact you distribute can’t run additional LLM calls on the viewer’s behalf.

    - LLM calls are the secret ingredient for magical mini-apps!

  - OpenAI’s GPTs allow setting a custom prompt and script for an LLM chatbot, allowing magical answers to users constrained by the author’s guardrails.

    - The fact the viewer pays for the LLM calls means the author doesn’t worry about going bankrupt from sharing the thing they made.

    - But you can’t do any structured UX other than chat.

  - What if you could have a magical mini-app that could be loaded by anyone with the link, that could do LLM-powered magic using paid for by the viewer?

- One style of cooking: pick what you want to eat and then get the ingredients.

  - Another approach: “what can I make with what I have”

    - Adjacent to an approach called “cooking with the seasons”

- If you have a very specific goal with an LLM you’ll have a bad time.

  - LLMs are great when the user only knows roughly what they want.

  - They're terrible and hard to direct when the user knows exactly what they want.

  - They squish out unpredictably as you try to pin them down.

  - If there’s only one answer and if it doesn’t get it perfectly it's hard to steer.

- Language understanding used to be extremely hard, but now it is a commodity.

  - The part that continues to be hard is fulfilling the intent now that you understand what it is.

- OpenAI and Anthropic had an underlying LLM model that was so good that they could slap on a demo level of UX and it was a viable product.

  - But they are not differentiating on UX ability, that is not their strength.

  - The fact the products have been so viable is a testament to the power of the model, not their UX ability.

- Anthropic’s Artifacts are effectively a hackathon level of UX sugar on top of the model.

  - And yet they are compelling and feel powerful: a good indication that there’s a There There.

  - But to do something interesting and differentiated with the artifact direction requires significant investment in UX differentiation and other enablers that are not related to the model itself.

  - To make artifacts that are not just throwaway requires a security and data model.

- If your artifacts live-update based on your data, it's scary.

  - What if my data squishes out later indirectly in a shared artifact in a way I didn't intend?

  - Instead, you can make it so an artifact's live state only changes when I choose to dip it back in my current data... and I can decide if I like it before publishing that change for others to see it.

- A pattern for data-aware artifacts, adopting the three-bucket pattern from my earlier [<u>Doorbell in the Jungle</u>](https://komoroske.com/doorbell-in-the-jungle):

  - 1\) A bucket for active artifacts.

    - These are artifacts that you have proactively chosen to have active.

    - They can show you if refreshing them would pull in new data from upstream sources.

      - This allows you to preview updates without actually cascading them through the system, and possibly squishing out in some shared downstream artifact.

    - But they don’t update by default unless you tell them to, preventing the chilling effect fear of “what happens if an artifact I shared updated in the future in some indirect way to share something embarrassing”

  - 2\) A bucket for archived artifacts.

    - Artifacts that you used to run, but aren’t currently running.

    - These are artifacts that you’ve “deactivated”.

    - Makes it less scary to deactivate an artifact, because it’s easy to find it again.

  - 3\) A bucket for suggested artifacts.

    - These are artifacts the system thinks you might find valuable.

    - They might be shown off to the right: easy to ignore if they aren’t a good match, but easy to see in your peripheral vision.

    - These artifacts would show a preview of what they’d look like applied to your data.

    - If you like one, you can “pin” it by moving it to your bucket of active artifacts.

    - Pinning is an extremely high intent act, an application of human judgment of quality that can be used to improve the quality of the overall system.

      - An artifact that another user decided to pin, all else equal, is way more likely to be useful to another user too than a random hallucinated artifact.

    - The self-steering metric to optimize is “maximize the absolute number of suggested artifacts that a given user chooses to pin”.

    - Because you can anonymously aggregate preferences of many users, this quality curve has a network effect; the quality gets stronger the more users use the system.

    - At first, the suggestions won’t be particularly good and people will actively ignore the suggestions bucket.

    - But over time as the quality improves, the suggestions bucket will be how more and more users get their tasks done.

    - A secondary use case that could theoretically grow to overshadow the primary use case of recipes conjured up on demand.

    - This is the differentiated upside of the system.

- Transformer models at their core are aiming for median responses.

  - So when you want to get dissonance and surprise, you need something to keep wrestling with the model to keep it off the default path, throwing it off its balance.

- Safely coordinating between different applications is a problem that filesystems have tried to tackle for decades.

  - The rate of innovation is very low.

    - One of the last “innovations” was in 2018 where Mac OS X will pop a system dialog before allowing an app to read sensitive folders like Downloads.

  - Shouldn’t there be some way of doing something with, I don’t know, a capability model or something?

  - Fuschia has a number of extremely interesting concepts around state and filesystems… but it’s a whole parallel universe app ecosystem that requires reinventing all of the apps we use today.

  - Perhaps it’s just not possible to clean up filesystems as they exist today, we’re at the top of the hill.

  - Maybe there’s a new hill that doesn’t have the same local maxima of basic file systems, but also doesn’t require a whole new universe of new large scale applications like Fuschia does?

  - Perhaps a pocket universe embedded in our current universe of apps that has different laws of physics, and can grow to become a whole universe in its own right?

- This week I learned that Marx had a term for the special qualities of a broader system.

  - He called this characteristic “general intellect”.

  - Marx thought of it like the accumulated knowhow of the people and processes that make up a system.

  - High-performing organizations produce something much bigger than the sum of their parts, a kind of magic.

    - It happens best where you have extremely high trust of a diverse set of people who are individually good at what they do.

    - You could view this as a kind of general intellect, too!

  - It’s a magical, powerful force, adjacent to the magic of the Radagast.

  - You could argue that the magic of an ecosystem is also this kind of general intellect.

  - Who gets to benefit from the value created by this magic?

  - The answer is typically “the employer” or “the aggregator”.

  - You could argue that aggregation is a technology to reify the creative magic of an ecosystem into a form that can be sold.

    - For it to work, you have to create a boundary to enclose the ecosystem to be able to extract from it.

  - Sometimes there’s a mismatch of members of the collective feeling ownership of this general intellect that then someone else tries to claim ownership over.

    - You can see this for example with the reaction of the Reddit community to Reddit’s monetization deals with AI companies.

    - Another example of the mismatch is Adobe’s AI TOS where creators thought of Adobe as a tool provider, and Adobe thought of the creators as part of an enclosed ecosystem.

    - You could call this dissonance a “copernican trauma”.

      - The power dynamic is backwards from what the user had thought it was–a traumatic discovery!

- I had a fascinating conversation this week with a friend who had read *The Institutional Revolution*.

  - A few riffs from the conversation.

  - Aristocracy was a very stable system for managing in a low-trust environment with high measurement costs..

    - Effectively everything was privatized.

    - Bloodlines were put into positions of power over long-running concerns.

    - Each person in charge had a lot to lose: they’d be disinherited.

    - This system was stable… but very hard to evolve (to say nothing of the massive inequality).

    - When things got easier to measure, it became possible to increase trust and a different system was possible that could evolve much better.

  - Two ways of looking at politics: Polis and Oikos.

    - Polis optimizes for the healthy functioning of the state.

    - Oikos optimizes for people who are more like you.

    - People who take a more Oikos perspective can agree with even people who are not working in *their* interest.

      - “I’d do the same if I were him.”

    - Polis is about following the rules for the good of the overall system.

      - Only people in the middle follow the rules.

      - The very poor and the very rich tend to be more about Oikos.

- “Late stage” means the finance bro vibe has infected it.

  - You see things like:

    - Hustle porn.

    - “*Obviously* it’s all about maximizing returns in the short term. What else could possibly matter?”

    - “Stopping to think about implications is for losers”

  - The tech industry has been increasingly infected by this vibe.

- WebSim is beautiful because the things it generates are so fleeting.

  - It feels like you're the only person who will see this particular thing.

- If you execute a very mature playbook (e.g. B2B Saas) the playbook is the same for all players, all that's special is the niche you chose... and there will likely be others in the same niche.

  - So winning is about good enough execution at the fastest velocity possible and nothing else.

  - But in environments without an established playbook, that is blue ocean, there’s much more than just good enough execution at the fastest velocity possible.

- LLMs have a hard time with Rust's borrow checker.

  - The semantics of the borrow checker are totally implicit, hard to reason about.

  - LLMs are much better where the grammar and logic is captured formally in the syntax.

- Hacking should be like hacking through the jungle.

  - Not just hacking for hacking's sake but to get to something.

  - Perhaps something that you don't yet know where or what it is.

- LLMs can get confused by the meandering path it took to the right answer in the conversation.

  - You need to continually prune the conversation history and reground it to remove those confusing bits.

  - Like squashing commits in git.

- A concrete artifact, even one that is wrong, is at least a schelling point for discussion.

  - It grounds the discussion in a specific thing.

  - As opposed to an amorphous conceptual blob that looks different to everyone discussing it.

    - In that situation, there’s much more room for disagreements that are totally illusory because of subtly different definitions.

- Last week I asserted that all else equal a user would choose the more private product.

  - But a friend pointed out that’s not nuanced enough.

  - In practice, people also pick which product to use based on the *expected rate of improvement,* as well as the current quality.

  - Products that are less private have more data to use to improve the product.

    - There’s more signal to aggregate and use to improve the quality of the system and prioritize fixes.

  - This could mean that a more private product would lose, all else equal, if users thought that the less-private product would improve faster.

  - However, in practice it’s possible to get enough data to use to improve the product within conservative differential privacy limits.

  - The vast majority of non-private tools collect many orders of magnitude more data than they need, “just in case”, and then do very little with it to actually improve the product.

  - All that data sitting there is just waiting for an MBA or finance bro to propose doing something with the data to make a quick buck.

- A great example of “change the laws of physics, new things result”: [<u>https://tailscale.com/blog/new-internet</u>](https://tailscale.com/blog/new-internet)

  - Avery is a wonderful writer!

- A [<u>tweet</u>](https://x.com/krishnanrohit/status/1817092305779835107?s=46&t=vBNSE90PNe9EWyCn-1hcLQ) I like from my friend Rohit:

  - "Just like mobile phones created casual gamers who far outnumbered the pros, we'll soon have "casual developers" who will far outnumber the pros."

- A [<u>tweet</u>](https://x.com/karpathy/status/1817414746595094672?s=51&t=vzxMKR4cS0gSwwdp_gsNCA) I like from (sadly, a person I do not know personally) Andrej:

  - "You write computer programs.

  - I conjure digital automations.

  - We are not the same."

- LLM literacy is a thing.

  - LLMs are not tools for lay people. They're expert tools.

  - They’re easy for anyone to pick up and quickly get surprisingly good answers.

  - But to wield them accurately and with consistently good results requires a lot of nuanced expertise.

  - Similar to how navigating Google back at the beginning required a kind of literacy called Google-fu.

  - The people who can wield LLMs most effectively understand what LLMs are good at… but also understand the specific problem domain vertical they’re asking the LLM about well enough to spot mistakes.

    - It’s less about helping you navigate a totally foreign space with detailed guidance, and more about automating answers you could have done yourself with enough time and patience.