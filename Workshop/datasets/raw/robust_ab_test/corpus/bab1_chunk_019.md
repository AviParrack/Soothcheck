# 7/22/24

- Simon Willison’s frame on LLMs: [<u>imitation intelligence</u>](https://simonwillison.net/2024/Jul/14/pycon/).

  - I love this frame!

  - Both in terms of Imitation meat.

    - Not quite the real thing, a bit off in a way that makes you a bit queasy

  - But also the process that it follows--imitating what it's seen before.

    - Intelligence entirely through imitation

  - Lots of people do imitation intelligence in general.

    - Not real reasoning, the facsimile of it.

    - "Why do you think that?" / "I don't know, that's what other people have always said!"

- Gordon’s concept of [<u>“Last principles thinking”</u>](https://x.com/gordonbrander/status/1790381450660835695)

  - When working with LLMs, think about the superficial, last principles first, then work backwards to the first principles.

  - LLMs are all about vibes of what they’ve seen, the superficial optics.

  - So if you want to control LLMs don’t try to engage at first principles, engage at the superficial optics first.

  - For example, if you’re trying to get LLMs to write code for little mini-apps, don’t try to teach it how to write code in your particular custom framework.

    - Instead, just have it write code in React, which it’s most familiar with.

    - Later, you can kludge in a fake React-shaped shim to interpret it to your underlying semantics.

  - Meet LLMs where they are.

  - The LLM's intermediate representation should be a format the LLM is maximally familiar with.

- The big stories have a kind of intrinsic, large-scale momentum.

  - You understand how you fit into the big stories better than everyone else.

    - Because you know your own story better than everyone else.

  - So draft off the big stories by telling people how you fit into them.

- There are a lot of people who don’t trust tech.

  - They are intrinsically distrustful of aggregators, even if they don’t know that word.

    - The tech industry, in a nutshell, to them: a massive aggregation of power combined with a “go fast by not thinking through the implications of our actions”

  - Having experience building technology does not require you to also fit into that default tech culture.

  - It’s possible to have one foot inside the industry (deep technical experience building scaled products) and also reject the default culture (“don’t think through the implications of your actions”).

    - One foot inside the system, one foot outside.

  - Such a perspective might be someone that most everyone could find reasonable and root for.

- Humans intuitively fear nuclear and chemical weapons more than traditional ones.

  - There’s just something so much more scary about an invisible thing that could hurt us without us realizing it.

  - Compared with conventional weapons, which are acutely dangerous, but if they’re in use you know.

- People intuitively fear their data flowing where they can’t see it.

  - If your data can flow invisibly and do some kind of diffuse, unknown harm to you in the future, that’s scary.

    - Data is unlike physical things in that it can be copied for ~free ~immediately without affecting the original, and transmitted for ~free ~instantly.

  - What if you could make data more concrete, more rivalrous.

    - More like a physical thing that you can reason about.

- It's kind of wild how much people care about their data flowing to LLMs more so than to a generic cloud services.

  - Generic cloud services could do whatever they want with your data… send it on to other companies, store it in insecure locations, sell it to data brokers.

  - But for some reason there’s a visceral fear over and above a generic cloud service.

  - Maybe it’s the same invisible fear that your data will somehow get absorbed into a future LLM (that is, the provider will train on it) and show up some how that harms or embarrasses you in the future.

  - A diffuse, invisble threat is more scary.

- Confidential compute is only part of the story.

  - Confidential compute moves the root of trust to the hardware.

  - But it also verifiably locks the cloud host (think Google, Amazon, Microsoft) out from peeking at what is happening.

  - But the risk factor that most people care about as a user is not so much the cloud host, but the *service provider*.

    - That is, the creator of the VM that the cloud host is running.

  - The service provider is the one who might plausibly send your data to a third party or store it recklessly.

    - The threat of the cloud provider peeking is way less important than the service provider and owner of the data doing some unscrupulous.

  - Even if one of the major model providers added Confidential Compute for their consumer app offering, it wouldn’t change that much.

  - The threat people worry about is the LLM provider training on their data, and Confidential Compute doesn’t really do anything about that.

  - The ideal is [<u>private cloud enclaves</u>](https://docs.google.com/document/d/1w1RbFtk2AB1QjrmPMr3BWcrhv6uJiYzhPLQd07DN2Bc/edit#heading=h.pipyx2w5mv99).

    - Confidential from the cloud host.

    - Private from the service provider.

    - Verifiable remotely.

- Hallucinated mini-apps are having a moment.

  - WebSim and Windows9x are both surprisingly compelling.

    - Doubly so because they start off looking like a weird toy (low expectations) which means when they do something useful it’s a mindblowing moment.

    - Capped downside, significant upside.

  - Anthropic Artifacts are just interface sugar, but they make the feedback loop immediate and help give a gradient of learning.

  - But hallucinated mini-apps today have two significant problems.

  - 1\) They can’t safely work with your data.

    - The hallucinated apps all start off with no data in them.

    - They also don’t have any sandboxing to speak of… you should be very wary about putting data into any of them.

  - 2\) They can’t compose with other mini-apps.

    - That means the ceiling of functionality is the biggest mini-app that can fit in an LLM’s understanding.

    - The LLM’s capability sets a ceiling on what can be done.

    - Instead, you want composition of other mini-apps, building blocks into a much bigger whole.

    - The sky would be the limit of functionality, but you’d need a way to reason about the safety and data of the composed whole.

    - With enough tokens in context and big enough models, the ceiling might get reasonably high, but never orders of magnitude higher.

    - The only way to get the ceiling to keep ratcheting up is to allow composition.

  - To do 1 and 2 you’d need a private, secure base to allow safe composition of your data.

    - For example, private cloud enclaves and information flow control.

- The filesystem is a shared, global mutable state--all of the worst practices for predictable, easy-to-reason-about systems.

  - And yet, how else could different apps coordinate?

  - What if you could add version control and branching to the filesystem?

  - It would get confusing quickly, but you could keep it focused and smaller scale.

- LLMs are pachinko machines that have paths for anything that any writing humans have done in the past.

  - But if there wasn't any in the training set, it has no idea.

  - It matches based on superficial similarity, not fundamental similarity.

  - If there are things that are similar, fundamentally, to your task in the training data, but not superficially, it will get confused and not know what to do.

- For prompting, curated examples are more important than a well-crafted complex prompt.

  - A complex prompt has to be interpreted, and can only include what you said.

  - Well curated examples captures the *vibe* of what you want, even the things you didn't think to call out.

  - With larger context windows, there’s more space for examples than before.

- Imagine a popular company does something that *superficially* looks like what you’re doing.

  - On a *fundamental* level it’s different and more limited.

  - But the optics are faster and easier to absorb by a busy observer than the fundamentals.

  - So now most of your customers will see the superficially similar competitor first.

  - To convince them to use your thing, you have to convince them why the superficial similarities are not important but the fundamental differences are.

  - A much steeper gradient of adoption!

- Private cloud enclaves are possible to construct today.

  - They require an expertise to know what you’re doing and that you’ve done it correctly.

  - But even when you’ve done it correctly, non-experts won’t necessarily realize the alchemical properties you’ve just created.

  - Ideally we’d move to a world where creating and verifying a private cloud enclave is captured in a repeatable playbook.

  - Easier for non-experts to reliably execute on, and easier for non-experts to verify.

- Psychological safety is required for unsafe thinking.

  - Funnily enough, people often think psychological safety creates *less* rigorous thinking.

  - No, psychological safety allows more rigorous thinking as a group, because it enables people to share disconfirming evidence, and that’s necessary for rigorous thinking.

- In a goldrush, everyone rushes in before it's clear if there's value to harvest.

  - A red ocean before it's even clear there's value in the ocean!

  - The worst of both worlds

- To get a large effect, one approach is a linear approach by going for the head.

  - "This one customer would be 20% of our value, let's go directly to them".

    - This is inherently a linear-at-best approach.

  - Another approach is to go with the smaller ones and surf upwards by creating a network effect.

    - Go for the easiest ones now in a way that makes it easier and easier to bring others on board in the future.

  - Seems backwards, but in many cases, momentum makes it easier to land more customers.

    - So starting with the easy ones and developing momentum makes it more likely to capture more value.

    - It feels backwards though!

    - “Why are you going after the small fish?”

- Everyone loves being the king of a hill.

  - A rivalrous thing: there can only be one king of a hill.

  - But you can create more hills for people to be kings of.

  - Now it's cheaper to generate e.g. video games with small markets, which are more hills to be king of.

- What is costly (and implicitly known to be costly) is used in style to signal quality.

  - But when that gets cheap, the style moves on.

  - There's a lagging edge though where a thing that used to be expensive becomes cheap but people still think it's expensive.

- "I do all my own writing, I never use an LLM"

  - Meaning comes from cost especially if the cost is unnecessary.

    - As Clay Shirky puts it, it's better to have your parents fly over to sing happy birthday to you than having a good singer's version of that song from a CD.

  - I’ve heard of people using LLMs to help write… and then putting in faux typos to make it look more authentic and hand-crafted.

  - LLMs make errors in reasoning, but they don’t do typos.

- A betrayal of someone doesn't just hurt the betrayed in that moment, it undermines their trust in all interactions in the past and all people they trust.

  - It is has toxic indirect effects.

- If a leader makes it clear (even implicitly) they only want confirmation bias, that's exactly what they'll get.

  - They'll think they're getting real information, but it's total slop.

  - The more that no one else tells them disconfirming evidence, the more that the incremental person will see it's dangerous to tell them, too, and will be even less likely to share.

  - A ratcheting supercritical state until reality crashes in, and that crash might kill you.

- If you're given a "confirm my idea is good" project by a lead who doesn't want disconfirming evidence, ask them before you start "what kind of information would change your mind?".

  - Because before you start, if they can't answer that question, then it's obvious even to themselves they're being disingenuous.

  - Now you have a roadmap of a line or reasoning to try to show why it's not a good idea, that they will be forced to accept if you can because previous them said it would be convincing.

- It's funny that LLMs are both creating more crap we have to cut through and also pretty good at cutting through the crap.

  - The meme of every email being an outline expanded to full prose by an LLM, and then on the receiving end reduced to the outline via an LLM.

- A toehold is like a savepoint for progress.

  - A sustainable position up from where you are that you can now use as a new base.

- in a combinatorial search space it's np-complete to find the right solutions.

  - So you need to use signals from previous use (were users implicitly happy) and also precompute interesting combinations before query time so they can be fast.

- Apps are cinder blocks.

  - You can break down apps into micro-particles of functionality.

  - Particles are sand.

  - Your AI should be able to create bespoke sand castles just for you by shaping particles.

- In 95, AOL was better than the web.

  - In 99, the web was wildly better than AOL.

- Just because it's going well doesn't mean it's enough.

  - It's necessary, but is it sufficient

- The "Look what you made me do" abuser dynamic.

  - The more powerful entity has some kind of power that is uncouth to use or that they don't want to want.

  - But then the less powerful entity does something, and that makes the more powerful entity say "This brings me no joy and I don't want to do this, but you've forced me to \[do this thing I wanted to do but I know I'm not supposed to want to want that exerts power over you\]".

  - Power might be for example lashing out or tearing something down.

  - It's disingenuous because they wanted to do it anyway, and no matter what the victim did, they were going to do it, even with a flimsy excuse.

  - They blame the victim for their own infraction.

    - It feels good to them to transfer the blame. "Yes, I did a bad thing, but only because (the victim) forced my hand"

  - This dynamic is becoming much more common in our political reality.

- LLMs are society-scale intuition crystallized into a form you can easily talk to.

- If AI gets really, really good, then it will fade back into the background.

  - A thing you can take for granted and thus becomes invisible.

  - Like internet connectivity and electricity.

- Some people are rock collectors.

  - I'm an insight collector.

  - Collecting interesting examples, seeing the potential in even rough rocks, tumbling a curated set of them together (collaboratively with others) to find and polish the gems.

- Imagine an organization that has hit a goldmine.

  - Every direction you look looks like an aggregator-style payday.

  - But that makes it harder to pick which direction to go in to coordinate around.

  - The danger becomes not that you don't create value but that you don't do a coherent direction.

  - A huge store of value combined with a fast-moving internal swarm could be chaos,

- When you do something people don't understand, do observers give you the benefit of the doubt?

  - If you do, you have trust.

  - If you’re a brand, you have brand equity.

- A test of how resilient a system is: give it input that's not one of the examples on the marketing page, how well does it work?

- For large complex product domains, the hard part is not building any particular component.

  - The hard part is ordering the work to be as close to continuously viable as possible.

- A kind of odd mental model for storing state in a system.

  - Pure functions that take inputs and produce outputs.

  - To store state, loop back function’s output to the inputs of a new instance of the function.

  - This model is not too dissimilar from how memory in chips works.

  - This model more clearly permits a forking, ever-branching space of possibility.

  - A [<u>branchial space</u>](https://mathworld.wolfram.com/BranchialSpace.html) of possibility, where the current state of the system is a multi-dimensional cut through the space.

  - But you could easily do another cut through the space.

- "nasty surprises" is "unhappy accidents"

  - Minimize nasty surprises: but also the *fear* of nasty surprises.

  - The fear is what prevents people from doing it, not the actual base rate of nasty surprises.

  - Ideally the fear perfectly correlates with actual base rate, but some UIs lull you into false sense of security, and some UIs are scarier than they actually should be.

- A cool thing that propagator networks can do: if you unconnect the data source, it just stops updating.

  - It’s like an escalator: if it’s unconnected, it just becomes stairs.

  - This is kind of nice in some cases.

  - Imagine a graph of lots of different nodes all connected.

    - For example, maybe a node that summarizes your searches in a system and feeds it to another synthesized ranker.

    - This could live update as you do more searches… but that’s a bit scary.

      - What if you do a bunch of searches for hemorrhoid cream, such that now downstream things also update with that recommendation.

      - Maybe those recommendations might be shown to other users. Embarrassing!

    - Spooky action at a distance.

    - The *fear* of that happening would chill experimentation.

  - But imagine if instead of it being live-updating, it only updated when you connected the network temporarily to your data.

  - You dip the network into your private data; it absorbs all the interesting intuition, and then you remove it to let it dry.

    - You can dip it again if you want, but if you don’t, it won’t change.

    - You don’t have to worry about spooky action at a difference.

- Previous examples of open aggregators: email and the web.

  - Those didn’t have obvious business models, because they were built with fixed-cost software.

    - It’s hard to charge for software that is written once and doesn’t change.

  - But private cloud enclaves must be paid for continuously by the user.

    - If the user didn’t pay for their turf, then they wouldn’t call the shots.

  - This implies a business model, unlike previous open aggregators.

- Anthropic made artifacts sharable before they made chats sharable..

  - But artifacts don’t have any stored state. Every person who loads one gets a blank slate of that artifact.

  - Anthropic’s actions say "the chat is not important, the artifact is"

  - Another thing you could say "the artifact is not important, what the artifact is used for is"

- I believe the idea I'm working on will work.

  - But this must be so.

  - If I didn't believe in it I wouldn't work on it!

- Turns out LLMs can [<u>help deprogram conspiracy beliefs</u>](https://x.com/DG_Rand/status/1775618798717911424?t=YBDF_8499LFUmM2jM12-qw&s=19) by being extremely patient and knowledgeable.

  - The asymmetry that it's easier to spread BS than to counteract it may have evened out a bit (at least for non-current events)