# 11/27/23

- *I was on vacation last week so these notes are primarily from the week before last.*

- When you have a magic black box at the core of your platform, disentangling it is a good idea even if it's not directly exposed to your users.

  - When you have a massive tangled bramble of magic and special cases, it's impossible to think about it in broad, coherent ways.

    - Every single feature is a special case; every incremental tweak or addition tends to add yet another special case.

  - When you have a conceptual model that captures all of the real world semantics of the system today (and also the expected semantics for the next few years) that's a game changer.

  - Now, incremental work can align itself with that model, cleaning up what pre-existing parts it touches, and laying down new aligned components.

  - Each incremental bit of work you do reduces the amount of overall magic, just a bit.

  - Over time, the system will naturally get more coherent, in an emergent way.

  - At some point a customer will ask "Hey can you provide us with a knob to do X", and they'll describe something very similar to how the internal system actually works.

  - At that point, it's a no-brainer to expose the knob (perhaps in the basement, not for non-savvy users), because you know it's a fundamental long-term semantic and not some random implementation detail.

- Beauty is in the eye of the beholder.

  - If you have a feature in your product you think is ugly, but your users tell you it's beautiful, then it's beautiful.

- "Works perfectly out of the box" is only possible for a small number of similar users.

  - With user diversity and/or scale it's not possible.

  - That's why the ideal is a "pre-assembled lego set".

- Spaghetti code will never be eradicated.

  - Spaghetti code accumulates where the value flows through in a system.

  - For the user, mucking with it is all downside and little upside.

  - Spaghetti code in a context tends to only ever accumulate.

  - You can't hope that a customer moves their spaghetti code over to your platform, or to create a clean platform where users won't need to ever do any spaghetti code.

  - The play is not to replace spaghetti code, it's to be the place where a customer's spaghetti code lives.

- A platform owner can't say "we simplified this for you" to a savvy customer.

  - It's OK to offer customers suggestions in case they think you're smarter and want your opinion.

  - But never assume that you're smarter and *force* them to take your suggestion.

- In simple circumstances, it's easy to tell if on net the value is greater than the cost.

  - In complicated circumstances (balancing lots of different sub-components of different types) it can be considerably more difficult.

  - In those situations, a thing that looks like a no-brainer might actually be a bad idea.

  - Situations that "should" be simple can sometimes be accidentally complicated; a kind of "gerrymandering".

  - Making it so the important decisions are easy to make well with few moving parts is a form of paying down debt.

- A recipe for expensive mediocrity: having to convince *everyone* it's a *good* idea.

  - Game-changing ideas are basically impossible in this situation.

    - A game-changing idea fundamentally will have at least some concrete downside from the status quo, and *might* have upside (and sometimes that upside turns out to be transformative).

  - A few places this dynamic shows up:

    - Large companies with bottom-up cultures

    - Peer review of academic papers

  - A better goal is: "no one thinks this will kill us and at least one person thinks it will be actively great"

  - In an environment that is amenable to experiments, this is a much better goal and more likely to find great things that work.

  - You can also invest time and effort to build guardrails to reduce the cost/danger of the average experiment, allowing you to do more of them.

- Most playbooks assume an environment where value is somewhat scarce.

  - In those typical situations, you want to consider a lot of options, carefully choose one, then focus on it.

  - But in environments where there is tons of value everywhere (e.g. after a general-purpose, game-changing technology becomes widely available), that advice is wrong.

  - In those cases, the danger isn't that you pick the wrong thing, it's that you don't pick anything.

  - In those cases, it's easy to get paralyzed figuring out which option will have the highest value.

    - It's like [<u>Buridan's ass</u>](https://en.wikipedia.org/wiki/Buridan%27s_ass), which had two equally good bales of hay that it couldn't decide between, so it died of starvation.

  - Part of the problem in these environments is that you don't know which option will have the highest expected value.

  - But no matter which option you pick, you'll almost certainly pick one that creates a lot of value.

- The hardest thing about getting things done is convincing other people.

  - Knowhow (intuitive, situated knowledge borne from experience) is extremely rich... but impossible to transmit to others at even a small fraction of fidelity.

  - If you have to convince many others before you can do anything, you'll do nothing.

  - It's liberating to be able to take small, experimental actions yourself.

  - Then, ideas that turned out to work become a no-brainer for others to double-down on once they're shown to be successful.

- It's easy to trick yourself when optimizing for metrics.

  - A real world example, a second hand story from Google Maps.

  - When viewing a place page, there's a button to get directions, with a big obvious icon.

  - The text beneath the icon used to be how long it would take to drive there, if you left right then: "15 minutes"

  - When experimenting they found, somewhat surprisingly, users seemed to like it better if the text instead said "Get Directions".

  - But this improvement was an illusion.

  - The "user satisfaction" proxy was CTR, and CTR did indeed go up when the text was changed to "Get Directions".

  - But that was not because users understood the button better.

  - Instead, it was because a class of users who previously had their needs met without clicking the button ("Do I need to leave now?") now had to click through to get their information.

- There are many different ways to spend time, with very different characteristics.

  - A quick 2x2 of two dimensions:

    - Personal vs Shared

    - Mundane vs Transcendent

  - Turbulent (Personal / Mundane)

    - The default state.

    - Random, reactive.

    - Never getting time to execute on any thread of creative effort for more than a moment, never knowing what's next.

    - Nothing of value emerges from this state.

  - Clockwork (Shared / Mundane)

    - The common state for adults in Serious Work.

    - Everyone synchronized to an external clock.

    - Everyone subverting their own rhythm to the centralized clock of the machine of society.

    - By being in sync, things that are impossible to do alone become possible.

    - Always knowing how many minutes you have until your next time-based commitment / meeting.

      - One eye on the clock, constantly.

    - Great for execution on plans.

    - Gives you a background metronome and structure to slot work into, a default forward momentum.

    - "I have 45 minutes to get A, B, and C done. I need to get started or I'll never get it done before the next meeting".

    - All of the synchronization and pinned down plans make it hard to adapt.

    - Many good things emerge from this state.

  - Flow (Personal / Transcendent)

    - Entirely in sync with your own internal creative energy and potential.

    - Disconnected from time and those around you, fully in flow state.

    - Only possible to get into this state when the muse strikes.

    - Being ripped out of this state feels like losing a limb.

    - All great things emerge from this state (or the next one).

  - [<u>Scenius</u>](https://www.wired.com/2008/06/scenius-or-comm/) (Shared / Transcendent)

    - Multiple members of a group in flow state *together*.

    - Lost from time and the surrounding system, but *together* in creative potential.

    - Massive ripple effects through time and space emerge from these acts of creation.

    - The highest possible state.

    - The most important things ever created come from this state.

  - Transitions between states are very, very hard.

    - The momentum of clockwork sometimes can give you a "kick" of energy so when the muse hits you you can go.

    - One way to make sure you have momentum when the muse hits is to have a bunch of pent-up ideas you're just waiting for some time and space to do, so when you finally get the time, you can just *go*.

  - The different states are more or less frequent. A ballpark guess of how many hours of humanity's time is spent in each:

    - Turbulence: 60%

    - Clockwork: 30%

    - Creative: 9.9%

    - Scenius: 0.1%

  - We can get better or worse at putting ourselves in the right state at the right time.

    - How can you get out of Turbulent state more often? How can you get more time for Creative?

    - How can we nurture the potential for Scenius?

- Alignment out of chaos is inordinately expensive.

  - You have to get everyone to point in the same direction *at the same time*.

  - The cost gets super-linearly more expensive with the size of the group to align.

  - When there is an existence proof in the world to point to, coordination gets orders of magnitude easier.

    - "We want to do something like *that,* but with these tweaks."

    - Everyone can see with their own eyes that something similar is viable.

- Alignment is easier to keep than to create.

  - Once you have some success with a group, it's easier to keep it going than to do something new.

  - Static friction is higher than rolling friction.

  - Once you get going, you can factor out the alignment into external structure to make it easier to stay aligned.

    - Process and infrastructure are examples of this structure.

    - A kind of [<u>stigmergy</u>](https://en.wikipedia.org/wiki/Stigmergy) for coordination.

  - You can think about processes like cached coordination results.

- A couple of weeks ago I wrote that systems that are in dynamic equilibrium can be on the cusp of phase transitions.

  - Phase transitions are transformative and game-changing.

  - As a person, when you're at peace, you're in a form of dynamic equilibrium.

  - Being at peace is being in balance, ready to move decisively into a new phase when the opportunity presents itself.

- From the outside it's hard to distinguish static equilibrium from dynamic equilibrium.

  - The former is approaching stasis: death.

  - The latter is poised to capitalize on game-changing opportunities when they present themselves.

  - Both can look almost lazy to outsiders, but the latter can create transformative value.

- Imagine the widest cone of results that could *possibly* come from an analysis you're considering doing.

  - What would you do differently in the short- to medium-term depending on those different extremes?

  - If the answer is "nothing", then don't do the analysis, just do the thing!

  - This is especially true if the downside risk of doing the thing is small and capped.

  - If it's a low cost no-brainer, the bar to clear should be very low.

  - The opportunity cost of analysis, of bringing every last person along, is in practice the largest component of execution cost in large organizations.

- If you dissect a system, you won't find the magic.

  - The magic doesn't come from any particular sub-component.

  - It emerges out of the connections between the parts.

  - Where's the magic? The whole thing!

  - This is one of the reasons that it's hard to analyze an ecosystem.

  - "This extremely popular SaaS platform with a big ecosystem is a crappy product that no one seems to like, and yet it has accelerating momentum. Where's the magic?"

- A consistently powerful source of alpha: humble curiosity.

  - Don't tear down what you hear from others; steelman it.

  - Not "what this person is missing is...", but "what this person is seeing is..."

- Optics are easy and obvious and quick.

  - Fundamentals are hard and nuanced and slow.

  - In the end, optics don't matter. Fundamentals are everything.

- Some subset of seedlings will grow into a mighty oak tree.

  - You don't know which ones will turn out to be mighty oaks, and which ones won't survive.

  - But they all start out as seedlings, and seedlings are cheap to nurture.

  - Seedlings just need some shade and patience.

  - As time goes on it will become increasingly clear which seedlings have the most potential, and doubling down on them will be a no-brainer.

- Good epistemic hygiene is not "write a lot of docs" and "do every analysis down to minute detail in triplicate."

  - It's "humbly seek disconfirming evidence and absorb it into your mental model, and don't waste time chasing down the illusion of certainty".

- It's easier to analyze a painting than a mirror.

  - People don't want to look too closely at their own imperfections; it's threatening and scary and they'd rather not do it.

  - A narrative trick: make a mirror that looks like a painting, so people can engage with it more deeply.

  - "Hey wait a second, this painting actually kind of reminds me of us!" / "Really? What a crazy random happenstance!"

- It's not possible to pick and choose the best characteristics of your peers and do them yourself.

  - Best practices have to be grown in place, over time.

  - You can't duct tape cultural practices willy nilly; they won't fit, and are often incompatible in non-obvious ways.

  - The only thing to do is to set a direction and over time arc towards it, making tradeoffs between incompatible components when necessary.

  - It's not possible to have all three of the broad bottom-up entrepreneurially of Amazon, the quality of Apple, and scale.

- A successful experiment is way more valuable than an intricate, grand doc.

  - A successful experiment is an existence proof of viability to double down on.

  - Don't judge teams by how many small, safe experiments fail.

  - Judge teams by how many experiments succeed and can be built upon.

  - Don't try to pre-judge which small, safe experiments will work out. Just do them.

- You have to be a realist about the things that might kill you.

  - You get to be an idealist about the parts that won't kill you.

- A roast turkey is really hard to cook well.

  - Even when it's cooked well, it's merely good, rarely *great*.

  - The vast, vast majority of turkey dinners are dry and bland.

  - There are a lot of situations in life like roasting a turkey: very hard to do well, and even if you do there isn't much upside.

  - If you aren't roasting the turkey for some other reason (connecting with family and society over a shared tradition) then maybe don't bother?

- When the system is mostly known and unchanging, you want factory farming.

  - That is, scale, efficiency.

  - When the system is mostly unknown and fast changing, you want community gardening.

  - That is, resilience at the level of the system and local experimentation.

- The technium is a society-scale computer; a sprawling, emergent system.

  - If you look for the magic in any individual component, you won't find it.

  - And yet the emergent result is extraordinary.

- I was talking with Max Kirby about the idea of seeing AI as a new type of fiber woven through the quilt of the tecnium.

  - He relayed a story about an octogenarian friend of his.

  - His friend doesn't own a smartphone or use technology.

  - When Max picked him up from the airport, Max navigated home using Google Maps on his phone.

  - His friend looked at the phone and asked, "Is this AI?"

  - Perhaps the answer isn't as straightforward as we think.

- A couple of weeks ago I observed that humans, computers, and pond scum are the same kind of thing.

  - That's kind of a bummer: we aren't special!

  - A few days ago my 4-year old told me she was sad because in a given situation she realized she wasn't special.

  - I told her: "It's OK, honey, in the grand scheme of things, none of us are. But that doesn't stop us from doing special things!"