# 6/17/24

- Last week I gave a guest lecture at an Oxford seminar on AI and Philosophy.

  - I had never been to Oxford before. What an extraordinary experience!

  - The title of my talk was [<u>Surfing the Silver Swarm</u>](https://komoroske.com/silver-swarm).

  - It wasn’t until I pulled the talk together that I realized that most of the bottom-up strategies I apply are intuitively about orchestrating the energy of a swarm and then surfing it.

  - The talk covers the asymmetric power of swarms, why they are so hard to control, and why if you let go of the need for control you can surf them to miraculous and surprisingly coherent outcomes.

  - Swarms are like flood-fills of possibility space, a search algorithm that scales linearly with investment.

    - With a single individual plan, the effectiveness typically scales sub-linearly.

  - The overall motion of the swarm is the sum of all of the motion vectors of its sub-entities.

    - By default, a random swarm has brownian motion, which means the net vector of the swarm is nothing.

    - But if you can induce a small asymmetry *consistently* across each agent, you can get significant movement of the overall swarm.

      - Interestingly, each individual agent in the swarm will (correctly!) report only a very small influence on them by that asymmetry.

      - But when you average all of the agents together, that small but consistent asymmetry adds up to something that strongly stands out from the noise.

    - It’s also possible to use a kind of macro-scale Maxwell’s Demon effect.

      - Pick the subset of entities in the swarm that are already going in the direction you want to go in, and select them into a subset.

      - The subset will now have a significant amount of velocity in roughly the direction you want to go in.

      - Now, the normal gravity-well dynamics of a successful swarm kicks off, and the energy becomes self-accelerating.

        - Others nearby who are mostly aligned will choose to join in, upping the momentum.

      - You’re effectively creating free energy by just applying some discerning judgment on what to allow in!

- Apple Intelligence is a continuation of a mode of app automation I think of as “fracking”.

  - For the OS to be able to understand what an app is doing, and to be able to poke deep into it to cause specific things to happen, the app has to make itself legible to the system.

  - Instead of a big monolith, the app has to create seams and sub-components, a process that is expensive and messy and I think of as “fracking”.

    - The process can also be somewhat extractive, as the system can now squeeze apps more tightly and use them as a commoditized component in a larger user flow the system controls.

  - Apple has been investing in this approach for more than a decade, all the way back to public NSUserActivity’s.

    - Each release of iOS they add more functionality for apps to frack themselves, more places that fracked apps can show up in the system.

  - Apple is not the only one doing this, by the way.

    - Google has been trying, although significantly less successfully–apps are far more wary about making themselves legible to Google.

  - This is not the only way to do this.

    - These approaches to app automation are with the app’s assistance, but it’s also possible to do an “over-the-top” style automation without the app being aware of what’s happening.

    - The “over the top” model is one that Arc Browser, Adept, Multi-ON, Anon, etc are all attempting.

    - The over-the-top model is the only viable approach to integrating with existing experiences for entities that don’t control the operating system.

    - The fracking approach starts with a high-quality set of apps that opt themselves in (depth over breadth).

    - The “over-the-top” approach starts with a low-quality but broad set of apps (breadth over depth).

    - Either approach might end up working!

  - Apple Intelligence is another incremental but significant step in the same direction they’ve established for a decade, now using the power of LLMs behind the scenes to increase their search quality.

  - This model can create user value.

    - But it has a low ceiling, because the app model presumes a privacy and distribution model that makes it:

    - 1\) challenging to launch an app speculatively in these kinds of assistive flows

      - Imagine asking Siri for help researching a trip to Hawaii. The next day you get an email from Delta trying to get you to buy tickets to Honolulu. As a user, that’s a “wait, *what*?” kind of moment–it’s not until after the fact you realize that behind the scenes the app was running and could have done who knows what.

    - 2\) impossible to run apps that the user hasn’t already installed.

  - As a result, the various methods for integrating with apps of today can only ever achieve accelerated copy paste.

    - An outcome that is potentially good enough, but never transformative.

  - An alternate approach is to precipitate a new kind of software with different laws of physics that are more naturally amenable to speculative, combinatorial assistive user flows.

- Both Confidential and Private Compute are about “provably locking observers out”, but for different observers.

  - Verifiably Private Compute says “The author of this service has locked themselves out from peeking inside, and as a user you can verify that externally.”

    - Authors can use tools like remote attestation to make this claim, attested to by the cloud host.

    - It allows users to trust that a cloud service is not, for example, logging their behavior.

    - This is what Apple has done with [<u>Private Cloud Compute</u>](https://security.apple.com/blog/private-cloud-compute/).

  - Confidential Compute says “the host of this service cannot peek inside, and as a user you can verify that externally”.

    - Host here means the cloud provider, e.g. AWS, GCP, Azure.

    - Confidential Compute uses hardware support in off-the-shelf chips to ensure that the VM is encrypted in memory.

      - This means that the host can’t peek inside running VMs.

    - Confidential Compute does not say, by default, that the author of the service isn’t peeking.

      - However, Confidential Computing can make claims of being Verifiably Private stronger.

      - For example, remote attestation claims from Confidential Compute instances are attested to by the hardware root of trust, instead of by the cloud host’s software layer.

    - Apple did *not* use Confidential Compute for Private Cloud Compute.

  - Private and Confidential Compute are both useful claims and they are complements.

    - The gold standard is both private *and* confidential; both guarantees of “can’t be peeked inside”.

  - Confidential Compute has been a slow and steady sea change, but mostly for specialized B2B applications (e.g. defense contractors).

  - Apple’s Private Computing is the first time that consumers have been told they can and should expect a higher standard of privacy for cloud computing.

  - The shift to private+confidential computing in the cloud seems inevitable!

  - This shift will be *extremely* hard for incumbents to successfully retrofit to their existing technical architectures and business models.

- Apple has made a "sustaining innovation" bet on AI.

  - And they (seem to have) executed very well on it!

  - A well-executed sustaining innovation play by one of the most powerful incumbents removes some of the oxygen in the room for disruptors.

    - This was a play that Microsoft did with browsers back in the day.

  - Apple has done the “simply cut out the edge cases” opinionated cut, which creates a radically simpler thing than alternatives that still delivers maybe 80% of the value.

  - But if AI has the potential to be disruptive, then that 80% of the value might actually be more like 10%; too little to actually stop the disruptive force.

  - If AI turns out to be a disruptive technology, the incumbents can only delay it, they can’t stop it.

- Emergent systems can absolutely create emergent cathedral-style outcomes.

  - Last week I talked about the Sea Shanty craze on TikTok.

  - A natural example is termite mounds.

  - Beautiful, monumental outcomes, without a central planner.

- A rule of thumb for when prompt injection might be a problem: if the model can call external tools *and* can accept untrusted inputs.

  - But as long as the model can’t call external tools, or the data comes only directly from a user or a trusted component, then prompt injection isn’t too big of a worry.

  - But note that although the latter is typically obvious when done directly, it can be very easy to do accidentally indirectly.

  - For example, if you use RAG in your pipeline to create a summary with an LLM with no tool use (safe) but then pass that summary on to a downstream LLM that allows tool use (potentially unsafe).

- LLMs can't be trusted with private data or data that might try to prompt inject them.

  - But imagine a set of tubes that by construction can only be combined in legal combinations that maintain integrity of data flows and confidentiality.

  - An LLM could absolutely construct a set of those tubes for data it can't see to flow through.

- Traditional consumer computing in cloud services (e.g. Gmail) is like staying on someone’s couch for free.

  - You’re a guest in their house. You abide by their rules, and they’re doing you a favor letting you be there in the first place.

  - Traditional cloud computing infrastructure (e.g. AWS) is like renting an apartment.

    - You have a landlord who owns the building, and could *theoretically* open your apartment and peek inside (and indeed they have the legal right), but it is reasonable to expect they wouldn’t do that except in exceptional or emergency situations.

    - You’re paying the landlord to host you, so your incentives are more aligned. An overly nosy landlord would not be a popular hosting provider.

  - Confidential computing infrastructure is like having an embassy in a foreign country.

    - Technically you’re embedded in someone else’s sovereign territory, but it's your own sovereign territory within it.

    - If someone breaks in, it’s an act of war.

    - Before Confidential Computing, the only way to have total sovereignty over your computation was to hold it in your hand.

      - The Roman saying of the person who controls something is the one who is allowed to destroy it.

      - But now that zone of sovereignty and trust can extend to remote servers with more compute, energy, and bandwidth than your phone.

      - And if you cleverly use remote attestation, you can assemble webs of trusted computation on untrusted compute nodes that you know work in a particular way, no matter who’s hosting them.

  - Confidential computing allows a radically different cloud computing paradigm.

- A few fun use cases for young kids and LLMs that some friends shared with me.

  - When driving somewhere with a kid in the car, ask ChatGPT, “Tell me about gas giants” and then help the kid ask follow-up questions.

  - You can also ask ChatGPT to come up with some quiz questions for your kid to answer.

  - LLMs are indefatigable conversation partners for curious young minds, especially for knowledge that is extremely common (and unlikely to be wrong).

  - LLMs also have deep knowledge of whatever random fandom obsession your child might have, for example if they want to dive into the minutiae of Star Wars lore, LLMs have read all of Wookieepedia!

- Systems have different affordances for forking and joining threads of exploration.

  - If a system makes forking easy but joining hard, then over time energy will dissipate from the system, as everything goes its own way.

  - If forking is easy and joining is also easy, you’ll get whatever emergent thing the complex system selects for… which might be glorious or might be horrible.

  - If forking is not easy and only joining is possible, you’ll get a singular vision, that might be able to cohere for some time, but will build up stresses as it drifts from its environment, which might explode catastrophically at some point.

- Every protection layer has holes.

  - It’s Swiss cheese.

  - But if you stack up enough layers then even if there’s a lot of holes in each layer, as long as they don’t line up and you can get good defense.

  - A good enough defense in practice built out of deeply imperfect components.

  - The smaller the average size of holes, and the more layers you stack, the more likely that the whole successfully defends.

- Ride the wave that is breaking now where you are.

  - Not the wave your plan said was supposed to break where you were supposed to be.

- The original mission statement of Apple: "To make a contribution to the world by making tools for the mind that advance humankind"

- There’s a difference between malleable and remixable.

  - Both imply experimentation and exploration.

  - Malleable implies changing the original.

  - Remix implies copying the original and tweaking.

  - Malleable software is OK if it's *your* software.

  - But if it's someone else's, they likely will get mad at you if you change it!

  - So when it comes to an open-ended system of explorable software, remixing is better.

- At any given moment in the race, it might look like the hare is winning.

  - But also at any given moment, they are more likely to die (or, less morbidly, to go to sleep and miss the rest of the race).

  - How far ahead they are is more obvious and visible than how likely they are to die.

  - That asymmetry of optics makes it look like they are winning when actually they might be more likely to lose.

  - The tortoise stays inside the adjacent possible.

  - They are more likely to win not because they are faster at any point, but because they are more likely to survive to the end.

- Continuing to do something is a different, lower threshold to clear than starting something.

- The universe’s shambolic but indefatigable path to success:

  - 1\) A massive swarm of things stumbling into the future.

  - 2\) Then the real world post-hoc selects the ones that turn out to be *fortuitous* stumbles

  - 3\) Then we narrativize the winners and completely forget the 99.99% of failures.

  - 4\) Then we repeat!

  - This emergent algorithm can create massive structures out of random starting conditions.

    - A little bit of random noise leads to a fortuitously-viable change that is conserved over multiple rounds.

    - The random noise now stands out a bit, distinct from the background noise.

    - The narrativization makes us as agents more likely to lean into the distinctiveness and accentuate it.

    - As long as those extensions are still viable, they also persist.

    - The more that the thing stands out from the background, the stronger the narrative gets, and the more that future extensions are likely to extend its throughline.

    - The result can be massive, durable structures that started off as random happenstance and blew up into a society-scale phenomena.

- The parts of your story that fit your narrative you’ll keep talking about.

  - The parts that don’t will fade away.

  - This creates the illusion of a very clear, directed path.

  - In reality it's an iterative process of retconning the past, finding the narrative throughline, and then leaning into that throughline going forward, which accentuates it.

- All lenses are lies.

  - They have to omit things to make other things pop out.

  - Omission of potentially pertinent details is a lie (at least a little bit).

  - But once you know that lenses are lies, you can still use them to find the truth.

  - Don't lock a lens to your face.

  - Do use a diversity of lenses to look at situations to see what truth pops out.

- We tend to tell ourselves self-serving lies.

  - It's extremely easy to lie to yourself!

  - Every simplification is, inherently and necessarily, a "lie" because it has to be to simplify.

  - Narrativizing is simplifying.

  - And you will pick the self-serving one without thinking about it, because that's the one that all else seems better to you.

- A differentiated way to create value: being able to pluck things just on the far edge of the adjacent possible.

  - You look magic to people who aren’t paying attention.

    - “He just plucked that out of thin air!”

  - When something goes from outside the adjacent possible to inside the adjacent possible, it's a discontinuity.

  - People only know something is on this side of the adjacent possible when they see an existence proof.

  - You can see through the adjacent possible just a bit.

    - It's like a frozen waterfall.

    - Crystal, warped, but things just beyond it you can see if you look carefully and have a pre-existing hypothesis to help you resolve otherwise ambiguous details in a coherent way.

  - Even better, when you can see an adjacency in a dimension others don’t even perceive then it looks even more magical when you pluck something through the frozen waterfall of the edge of the adjacent possible.

- When someone gets rewarded or punished, everyone watching looks at "how does this person differ from their average peer" and then assume that every aspect of that difference was the reason why they were rewarded or punished.

  - Even if it wasn't the reason they got rewarded or punished!

  - That (possibly erroneous) belief can become self-sustaining.

  - People will avoid doing the thing that apparently is punished, and people will notice that other people don't do it, which will make it more likely others don't do it, too.

    - If no one else is doing it, if you do it, you *really* stick your neck out. Why chance it?

    - Similar logic to superstitions on professional sporting teams.

  - In a dysfunctional environment, a very small inciting incident can reverberate around for years.

    - Like a traffic jam that persists for hours after the original obstruction is cleared.

- Vertical products start with their median customer.

  - New horizontal platforms (e.g. mediums) start with a weed species unlike their median users to kickstart the ecosystem.

  - Vertical products have a singular network effect of strength in numbers.

  - Horizontal platforms have multi-layered, multi-dimensional feedback loops, which scale with the combinatorial possibility of the building blocks.

    - They scale much faster, allowing the platform to continually ratchet up the kinds of things it can accomplish.

  - Vertical products just get better and better at their vertical niche.

    - They have to be expanded horizontally through direct, intentional effort.

- LLMs don’t do deep reasoning.

  - They do superficial detail matching crazy well.

  - But it turns out that a huge number of superficial details, if generated by an underlying deep generative structure, have a deep consistency.

  - At large enough scales, if you average out all of the details, the noise falls away and all that’s left is the deep consistency of the underlying generative function.

  - The best way to compress the superficial details is to (indirectly) distill the generative function.

  - This makes LLMs very good at doing the appearance of deep reasoning; the ability to generate new superficial details that are consistent with the underlying generative system.

  - The LLM is good at this even if it doesn’t “understand” it.

  - The LLM has the vibe of the fundamental societal generative function at its core.

- Scaffolding can help you have better insights more often.

  - But the scaffolding doesn’t think for you, it just helps lever your thinking.

  - The framework doesn't think for you, it just gives you a structure to think within.

  - Nucleation sites for your intuition.

- A useless tool: a power drill with a drill bit made of playdoh.

  - Credit to my friend Ben Mathes for this one!

- Aggregators often apply a “gray” pattern that is only available to aggregators.

  - Because all consumer access must flow through the aggregator, they can “front run” the query stream.

  - If the aggregator has a great 1P answer, they just return it immediately. If they don’t they fall back on the 3P answer.

  - But even when they fall back, they can note patterns in the query stream to find pockets of important, adjacent use cases that they can build directly into their 1P solution.

    - Meanwhile, the 3P only sees an ever-shrinking subset of the query stream.

  - Over time, the aggregator sends less and less traffic to the 3Ps.

  - The aggregator can surf a wave of query traffic only they can fully see.

  - Sometimes at the start the 1P solution is way less powerful than the 3P solution.

  - The 3P solution might think they’re being clever by partnering with the aggregator.

  - But if the aggregator is even minimally competent, they will get better and better and cut the partner out more and more.

  - As a partner it’s like picking up pennies in front of a steamroller.

- A default stance: Assume everyone is good at what they do and doing what they think is best.

  - That way you can learn.

  - As soon as you think to yourself “They’re an asshole” or "They're an idiot" it terminates analysis.

  - The feedback loop to learn is snipped and stops being a learning loop.

- Coevolution happens because the entities don’t make a single decision but incremental, continuous, and interdependent ones.

  - A small move, then wait and react, then repeat.

  - This allows coordination, call and responses, and mutual reactions.

  - If everyone made their plans individually and then executed them without looking at what anyone else did, nothing emergently coherent could show up, and it would likely be a chaotic mishmash of conflicting things.

- One reason gossip is intriguing is because it’s disconfirming evidence.

  - Things people don’t say in polite company, some subset of which are true.

  - Disconfirming evidence is good for us, and yet we often steer clear of it because it’s challenging. But the more you can lean into it, the more you can get stronger in that environment.

- Fuzzy things can’t have finely drawn boundaries.

  - If you insist on a finely drawn boundary, you'll get a boundary of infinite fractal complexity, which will have infinite length.

    - The shoreline paradox.

- Evolution works across a long feedback loop integrating approximately over all variance in that loop.

  - All actions that happen between the organism being conceived, and when it successfully reproduces.

  - That’s an insane amount of random variation and stuff on that feedback loop.

  - And yet evolution is able to pick out the smallest changes to select for over time. How?

  - Even if it's an indirect and subtle effect, if it’s *consistent* then with enough instances the signal pops out against the background noise.

  - Even long feedback loops, as long as the signal is threaded through like beads on the string, it can work.

    - A small but consistently different alignment of beads on the string.

  - In one feedback loop iteration the signal is washed out in the noise.

  - But in thousands of loops for thousands of individuals, that consistent asymmetry will stand out against the background noise of all of the other ones averaged together.

  - This is one of the reasons deep learning training works so well despite insanely long and multi-varying feedback loops as signals backpropagate.

  - Just jam an unreasonable amount of data through it, and all of the noise is filtered out and the signal remains.

- Deep learning isn’t how the brain works.

  - It’s a brute force technique to get a thing to work that is viable.

    - It requires massive scales and cost,

    - It’s not the bottom up way for it to have shown up, e.g. in actual organisms.

  - Instead of finding the delicate, contingent, bottom-up path to deep-learning style outcomes, we just said "screw it we'll run it not a 9 or 10 or even 11, but 1000x power until it works".

  - Now that we know the basic thing is viable, there will be any number of optimizations and improvements to it that can be made.

  - The brain can get similar outcomes more efficiently, with less feedback. But it was an evolved, bottom-up, continuously viable machine, not a brute-force engineering approach.

  - There’s an existence proof of a similar scale system that runs far more efficiently (our brains); from here we can continually improve the artificial, brute force approach.

- Systems on the cusp of criticality are where interesting things happen.

  - Similar to the nested chain mail of feedback loops, always spinning.

  - Ready to be perturbed and cascade that information; a dynamic equilibrium.

- Observation is inherently noisy.

  - There's all kinds of details that don't matter--that average out to mush--that you can safely ignore.

  - But deciding which details matter and which can be ignored is an act of judgment.

  - It requires having a mental model of what you expect, so you can see the things that don't fit the model (and are thus important to attend to).

  - This is one reason why observing the whole system from afar is not viable; when you're up close you can sort through the signal, and then flag anomalies up the chain.

  - But when you're looking at all of it at once, you're drowning in a cacophony of signal.

  - If you don’t have a mental model for what you expect to see in the signal, it’s all just noise.

- All things die and yet life goes on.

- Alignment of agents creates brittleness

  - Agents align with other agents in the swarm, because it's easier to go with the flow.

  - This increases efficiency overall (fewer eddy currents, more laminar flow), but like any increase in efficiency, it makes the system better at doing what it currently does... at the cost of making it less able to change what it does.

- If the individual can’t survive outside the swarm they will become more similar to others in the swarm.

  - Any individual member of the swarm will increasingly look like any other.

  - Consider two red blood cells; they’re practically identical.

  - Once you submit to the swarm and its logic, the power of the collective, the individual has a crutch: they no longer need to be viable as an individual.

  - That allows them to exist more cheaply, to shed lots of complex machinery and effort, but it also means that they can no longer exist outside the swarm.

  - Ants become extremely similar because they aren't viable outside of the swarm.

  - The swarm would prefer the individuals to be replaceable, the same, cogs in the machine, because then it can handle them with more efficiency.

- On a traditional local OS you can change the application you use to access your data.

  - But in a cloud service or app you can’t.

  - Your data and the app are tied, fundamentally.

  - This reduces competition significantly; an app that has useful data in it (but not particularly good at making use of that data) will be hard for even better competitors to compete with.

  - Data silos are powerful in single-player contexts.

  - But they are even more powerful in multi-player contexts.

  - To switch, you have to convince everyone else not just to move to a new app, but to move to the *same* new app. A massive coordination problem, even if there’s a clear value differential!

- Even dialogue with yourself creates insights.

  - The process of distilling a thought dart forces you to reify vibes into words, which can force them to be more rigorous.

  - As you converse, there’s an opportunity to learn and change what you think based on prior insights in the conversation.

  - This is also one of the reasons that the same LLM, structured into agents that converse with each other to create a good answer, can get better results than that same LLM asked the question directly.